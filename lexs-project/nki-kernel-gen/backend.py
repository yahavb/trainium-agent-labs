#!/usr/bin/env python3
"""Run on the Trainium host: python backend.py (prints an ngrok endpoint)."""
import argparse
import ast
import asyncio
from contextlib import asynccontextmanager
import hmac
import json
import os
from pathlib import Path
import re
import secrets
import signal
import sys
import time
from typing import Literal
import uuid

from fastapi import APIRouter, Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
import psutil

ROOT = Path(__file__).resolve().parent
MODELS = ("gpt-6-luna", "gpt-6-sol", "gpt-6-astra")
ACTIVE = {"queued", "running", "cancelling"}


class DescriptionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    description: str = Field(min_length=10, max_length=20_000)


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reference_code: str = Field(min_length=1, max_length=100_000)
    initial_kernel_code: str | None = Field(default=None, max_length=100_000)
    model: Literal["gpt-6-luna", "gpt-6-sol", "gpt-6-astra"] = "gpt-6-luna"
    module_name: str = Field(default="custom-kernel", pattern=r"^[a-zA-Z0-9_-]{1,80}$")
    iterations: int = Field(default=10, ge=1, le=500)
    timeout: float = Field(default=600, ge=1, le=3600)
    wandb: bool = True
    verified: Literal[True]  # Caller explicitly approves the reference before execution.


def validate_source(source, entrypoints):
    """Syntax/contract check only. Uploaded Python is trusted executable code."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise HTTPException(422, f"Python syntax error on line {exc.lineno}: {exc.msg}") from None
    functions = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    missing = set(entrypoints) - functions
    if missing:
        raise HTTPException(422, f"Source must define: {', '.join(sorted(missing))}")


def redact(text):
    for name in ("OPENAI_KEY", "OPENAI_API_KEY", "WANDB_KEY", "WANDB_API_KEY",
                 "NGROK_AUTHTOKEN", "NKIEVOLVE_API_TOKEN"):
        value = os.environ.get(name)
        if value:
            text = text.replace(value, "[REDACTED]")
    return re.sub(r"\b(?:sk-[A-Za-z0-9_-]{12,}|wandb_v1_[A-Za-z0-9_-]{12,})", "[REDACTED]", text)


def read_json(path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def kill_tree(process):
    """Also kill evaluator workers, which create their own process sessions."""
    try:
        parent = psutil.Process(process.pid)
        children = parent.children(recursive=True)
    except psutil.NoSuchProcess:
        children = []
    for child in reversed(children):
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def create_app(root=ROOT, jobs_dir=None):
    root = Path(root).resolve()
    directory = Path(jobs_dir or os.environ.get("NKIEVOLVE_JOBS_DIR", root / "outputs/api")).resolve()
    jobs = {}
    tasks = {}
    generation_lock = asyncio.Lock()

    def save(job):
        path = directory / job["id"] / "job.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(job, indent=2))
        temporary.replace(path)

    @asynccontextmanager
    async def lifespan(app):
        token = os.environ.get("NKIEVOLVE_API_TOKEN", "")
        if len(token) < 24:
            raise RuntimeError("Set NKIEVOLVE_API_TOKEN to at least 24 characters, or start with python backend.py")
        app.state.token = token
        directory.mkdir(parents=True, exist_ok=True)
        for path in directory.glob("*/job.json"):
            job = read_json(path)
            if not isinstance(job, dict) or job.get("id") != path.parent.name:
                continue
            if job.get("status") in ACTIVE:
                job.update(status="interrupted", finished_at=time.time(), error="Backend restarted during this run")
                save(job)
            jobs[job["id"]] = job
        yield
        for task in tasks.values():
            task.cancel()
        if tasks:
            await asyncio.gather(*list(tasks.values()), return_exceptions=True)

    security = HTTPBearer(auto_error=False)

    async def authenticate(credentials: HTTPAuthorizationCredentials | None = Depends(security)):
        if not credentials or not hmac.compare_digest(credentials.credentials, app.state.token):
            raise HTTPException(401, "Invalid API token", headers={"WWW-Authenticate": "Bearer"})

    app = FastAPI(title="NKIEvolve API", version="0.1.0", lifespan=lifespan)
    api = APIRouter(dependencies=[Depends(authenticate)])
    origins = [x.strip() for x in os.environ.get("NKIEVOLVE_ALLOWED_ORIGINS", "").split(",") if x.strip()]
    if origins:
        app.add_middleware(CORSMiddleware, allow_origins=origins,
                           allow_methods=["GET", "POST"],
                           allow_headers=["Authorization", "Content-Type", "ngrok-skip-browser-warning"])

    def get_job(job_id):
        if job_id not in jobs:
            raise HTTPException(404, "Unknown job")
        return jobs[job_id]

    async def execute(job):
        process = None
        spawning = None
        path = directory / job["id"]
        try:
            job.update(status="running", started_at=time.time())
            save(job)
            env = os.environ.copy()
            # Tunnel and API credentials have no role in the evolution subprocess.
            for name in ("NGROK_AUTHTOKEN", "NKIEVOLVE_API_TOKEN"):
                env.pop(name, None)
            env["PYTHONUNBUFFERED"] = "1"
            spawning = asyncio.create_task(asyncio.create_subprocess_exec(
                sys.executable, "-u", str(root / "nki_evolve.py"), "--config", str(path / "config.json"),
                cwd=root, env=env, stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT, start_new_session=True, limit=1024 * 1024))
            process = await asyncio.shield(spawning)
            with (path / "console.log").open("w") as log:
                while line := await process.stdout.readline():
                    log.write(redact(line.decode("utf-8", errors="replace")))
                    log.flush()
            result = await process.wait()
            job.update(status="completed" if result == 0 else "failed", returncode=result)
            if result:
                job["error"] = "Evolution failed; inspect the logs endpoint"
        except asyncio.CancelledError:
            job["status"] = "cancelled"
        except Exception as exc:
            job.update(status="failed", error=redact(str(exc)))
        finally:
            job["finished_at"] = time.time()
            save(job)
            # If cancelled during spawn, still acquire and clean up the child.
            async def cleanup():
                child = process or (await spawning if spawning else None)
                if child:
                    kill_tree(child)
                    await child.wait()
            await asyncio.shield(asyncio.create_task(cleanup()))

    @api.post("/references/generate")
    async def generate_reference(request: DescriptionRequest):
        key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_KEY")
        if not key:
            raise HTTPException(503, "Set OPENAI_API_KEY or OPENAI_KEY on the worker")
        if generation_lock.locked():
            raise HTTPException(409, "A reference is already being generated; please wait")
        async with generation_lock:
            try:
                from openai import AsyncOpenAI
                async with AsyncOpenAI(api_key=key, timeout=180, max_retries=1) as client:
                    response = await client.chat.completions.create(
                        model="gpt-6-luna", max_completion_tokens=8192,
                        messages=[{"role": "system", "content":
                            "Write a self-contained CPU PyTorch reference module for an AWS Trainium NKI kernel search. "
                            "Return Python source only. Define reference(*inputs) and cases(seed), a generator "
                            "yielding positional tuples of CPU torch tensors and optional static arguments. "
                            "Use seeded generators. Include typical shapes, partition/tile tails, zeros and "
                            "operation-specific numerical edge cases. Outputs must be tensors or tuples/lists "
                            "of tensors. Use float32 or float16, never bfloat16. Set RTOL and ATOL explicitly. "
                            "Document input/output shapes and any assumptions in the module docstring. "
                            "No files, network, environment access, subprocesses, external data, NKI code or "
                            "top-level execution. Keep cases small enough to verify on CPU. The user will "
                            "review this source before it executes. Follow the described operation exactly."},
                                  {"role": "user", "content": request.description}])
                code = response.choices[0].message.content or ""
                fenced = re.search(r"```(?:python)?\s*\n(.*?)```", code, re.S)
                code = (fenced.group(1) if fenced else code).strip() + "\n"
                validate_source(code, {"reference", "cases"})
                return {"reference_code": code, "model": "gpt-6-luna", "requires_verification": True}
            except HTTPException as exc:
                raise HTTPException(502, "Generated reference did not satisfy the task contract: " + str(exc.detail)) from None
            except Exception as exc:
                raise HTTPException(502, "Reference generation failed: " + redact(str(exc))) from None

    @api.get("/health")
    async def health():
        return {"status": "ok", "models": MODELS, "active_job": next(
            (j["id"] for j in jobs.values() if j["status"] in ACTIVE), None)}

    @api.post("/runs", status_code=202)
    async def launch(request: RunRequest):
        validate_source(request.reference_code, {"reference", "cases"})
        if request.initial_kernel_code:
            validate_source(request.initial_kernel_code, {"kernel"})
            if any(marker not in request.initial_kernel_code for marker in ("EVOLVE-BLOCK-START", "EVOLVE-BLOCK-END")):
                raise HTTPException(422, "Initial kernel needs EVOLVE-BLOCK-START and EVOLVE-BLOCK-END markers")
        if any(j["status"] in ACTIVE for j in jobs.values()):
            raise HTTPException(409, "A run is already using this worker")
        if not (os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_KEY")):
            raise HTTPException(503, "Set OPENAI_API_KEY or OPENAI_KEY on the worker")
        job_id = uuid.uuid4().hex
        path = directory / job_id
        path.mkdir(parents=True)
        (path / "reference.py").write_text(request.reference_code)
        config = {"reference": "reference.py", "module_name": request.module_name,
                  "model": request.model, "iterations": request.iterations,
                  "timeout": request.timeout, "mode": "hardware", "output": "artifacts",
                  "wandb": {"enabled": request.wandb, "mode": "online",
                            "project": os.environ.get("WANDB_PROJECT", "trainium-kernels")}}
        if os.environ.get("WANDB_ENTITY"):
            config["wandb"]["entity"] = os.environ["WANDB_ENTITY"]
        # Reuse the repository's configured NKI authoring context, never its output paths.
        configured = read_json(root / "config.json", {})
        config["context_files"] = [str((root / p).resolve()) for p in configured.get("context_files", [])]
        if request.initial_kernel_code:
            (path / "initial_kernel.py").write_text(request.initial_kernel_code)
            config["initial_kernel"] = "initial_kernel.py"
        (path / "config.json").write_text(json.dumps(config, indent=2))
        job = {"id": job_id, "status": "queued", "model": request.model,
               "module_name": request.module_name, "iterations": request.iterations,
               "created_at": time.time()}
        jobs[job_id] = job
        save(job)
        task = asyncio.create_task(execute(job))
        tasks[job_id] = task
        task.add_done_callback(lambda _: tasks.pop(job_id, None))
        return job.copy()

    @api.get("/runs")
    async def list_runs():
        return sorted(jobs.values(), key=lambda j: j["created_at"], reverse=True)

    @api.get("/runs/{job_id}")
    async def status(job_id: str):
        job = get_job(job_id)
        path = directory / job_id / "artifacts"
        rows = []
        metrics = path / "metrics.jsonl"
        if metrics.exists():
            for line in metrics.read_text().splitlines():
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    pass  # Writer may be midway through the final JSONL row.
        return {**job, "metrics": rows, "wandb": read_json(path / "wandb_run.json"),
                "kernel_available": (path / "best_kernel.py").exists()}

    @api.get("/runs/{job_id}/logs", response_class=PlainTextResponse)
    async def logs(job_id: str):
        get_job(job_id)
        path = directory / job_id / "console.log"
        if not path.exists():
            return ""
        with path.open("rb") as stream:
            stream.seek(max(0, path.stat().st_size - 128_000))
            return redact(stream.read().decode("utf-8", errors="replace"))

    @api.get("/runs/{job_id}/kernel", response_class=PlainTextResponse)
    async def kernel(job_id: str):
        get_job(job_id)
        path = directory / job_id / "artifacts/best_kernel.py"
        if not path.exists():
            raise HTTPException(404, "No measured, correct kernel has been saved yet")
        return PlainTextResponse(path.read_text(), headers={"Content-Disposition": 'attachment; filename="best_kernel.py"'})

    @api.post("/runs/{job_id}/cancel", status_code=202)
    async def cancel(job_id: str):
        job = get_job(job_id)
        task = tasks.get(job_id)
        if task and not task.done():
            job["status"] = "cancelling"
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            # Cancellation before execute() starts has no finally block.
            job.update(status="cancelled", finished_at=time.time())
            save(job)
        return job.copy()

    app.include_router(api)
    frontend = root.parent / "frontend"
    if frontend.is_dir():
        # Only static UI assets are public; every API route still requires a token.
        app.mount("/", StaticFiles(directory=frontend, html=True), name="frontend")
    return app


app = create_app()


async def serve(port, local_only):
    import uvicorn
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, access_log=False))
    task = asyncio.create_task(server.serve())
    listener = None
    try:
        while not server.started:
            if task.done():
                await task
                raise RuntimeError("API failed to start")
            await asyncio.sleep(0.05)
        url = f"http://127.0.0.1:{port}"
        if not local_only:
            import ngrok
            listener = await ngrok.forward(f"127.0.0.1:{port}", authtoken_from_env=True)
            url = listener.url()
        print(f"\nNKIEvolve endpoint / frontend: {url}\nAPI documentation: {url}/docs\n"
              f"Authenticate API calls with Authorization: Bearer {app.state.token}\n"
              "Keep this process running. Ctrl+C stops the tunnel and active run.\n", flush=True)
        await task
    finally:
        if listener:
            await listener.close()
        server.should_exit = True
        await asyncio.gather(task, return_exceptions=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--local-only", action="store_true", help="Skip ngrok; listen on loopback")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    if not args.local_only and not os.environ.get("NGROK_AUTHTOKEN"):
        parser.error("Export NGROK_AUTHTOKEN before starting, or use --local-only")
    if not os.environ.get("NKIEVOLVE_API_TOKEN"):
        os.environ["NKIEVOLVE_API_TOKEN"] = secrets.token_urlsafe(32)
        print("Generated API token (copy into your client): " + os.environ["NKIEVOLVE_API_TOKEN"], flush=True)
    if len(os.environ["NKIEVOLVE_API_TOKEN"]) < 24:
        parser.error("NKIEVOLVE_API_TOKEN must be at least 24 characters")
    try:
        asyncio.run(serve(args.port, args.local_only))
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print("Startup failed: " + redact(str(exc)), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
