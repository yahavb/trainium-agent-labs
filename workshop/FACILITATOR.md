# Running the cluster for 100 participants

Each participant gets **one seat number** and **one pod**, `seat-<number>`, in the `default` namespace of
one EKS cluster. They reach it from their own laptop with the AWS CLI and kubectl. The participant side is
Part 1 of the [main README](../README.md); this page is the organiser side.

## The access model, and what it does not protect

Everyone shares **one 12-hour credential set** for the Isengard role `EKSExec`, copied from the Isengard
console. An EKS access entry maps that role to a Kubernetes group whose only rights are:

| EKSExec can | EKSExec cannot |
|---|---|
| list and get every pod in `default` | delete, create, scale or edit anything |
| exec into and read logs of `seat-1` … `seat-N` | exec into `qwen3-8b` or any other non-seat pod |

Its IAM policy (set in Isengard) needs `eks:DescribeCluster` on the cluster — that is all
`aws eks update-kubeconfig` uses.

**The seat assignment is an honour system.** The credentials are shared, so Kubernetes sees one user and
the audit log cannot tell participants apart. Say so at the start: *your seat is yours; going into
someone else's is the one rule.*

## Before the day

With **your admin credentials**, once:

```bash
export CLUSTER=hack-hyd AWS_REGION=ap-south-2
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)

# 1. Let EKSExec into the cluster, as the group workshop-participants
aws eks create-access-entry --cluster-name "$CLUSTER" \
  --principal-arn "arn:aws:iam::$ACCOUNT:role/EKSExec" --kubernetes-groups workshop-participants

# 2. Limit that group to listing pods and exec/logs on seat-* pods
kubectl apply -f k8s/workshop-rbac.yaml

# 3. Start the 105 seats, and wait for every one to show 1/1 READY
kubectl apply -f k8s/workshop-seats.yaml
kubectl get pods -l app=seat -w
```

Step 1 needs the cluster's authentication mode to be `API` or `API_AND_CONFIG_MAP`
(`aws eks describe-cluster --name "$CLUSTER" --query cluster.accessConfig`).

* **Capacity.** One seat = one chip (`s-lnc2`) + 11 CPU + 120 GiB RAM + 100 GiB disk. A `trn2.48xlarge` holds 16 seats, so 105
  seats need 7 of them. Pods that stay `Pending` mean the nodegroup is short.
* **The repo must be publicly cloneable** — the init container clones it without credentials. If it is
  still private, every seat sticks at `Init:Error`.
* **Weights.** First start downloads 16 GB per node into `/opt/workshop/hf-cache`, then compiles; allow
  10+ minutes with 16 pods per node starting at once. Bring the seats up the evening before.
* **Test the participant path yourself**: fresh terminal, paste only the EKSExec block, follow README
  Part 1. `aws sts get-caller-identity` must show `assumed-role/EKSExec`.

## Who takes which pod

1. **At check-in**, each participant gets a seat number 1–100, written on the sign-in sheet next to
   their name and on their badge. That number is their pod: seat 42 → `seat-42`.
2. **Seats 101–105 are spares.** Nobody starts on one.
3. **A broken seat** (stuck, crashed, wedged chip): give the participant a spare and note it on the
   sheet. Don't make them wait for a restart.
4. **To recycle a broken seat**, `kubectl delete pod seat-N` with your own credentials. The StatefulSet
   recreates it with the same name. **Everything in its `/workspace` is lost**, and the model takes
   ~5 minutes to come back. Tell participants to `git push` their work somewhere.
5. **Teams** of 3–5 each use their own seats and share code via git. A team can also share one seat
   by all exec-ing into it, but then they share one model and one chip.

## Sharing the credentials

* **Never commit them or push them to GitHub.** GitHub scans pushes for AWS keys and reports them to AWS;
  the keys get revoked or flagged as an exposure, and the repo is public, and git history keeps them.
* At the start of the day, copy the EKSExec block (12 hours) from Isengard — all three versions — and
  **post it pinned in the workshop chat channel**. Cluster (`hack-hyd`) and region (`ap-south-2`) are
  already in the README, so the block needs nothing added.
* Minted at 08:00, they last until 20:00. If the day runs longer, mint a fresh set and re-post; participants
  paste it and carry on — their pod and work are unaffected.

## After the day

```bash
kubectl scale statefulset seat --replicas 0
kubectl delete role,rolebinding workshop-participant
aws eks delete-access-entry --cluster-name "$CLUSTER" \
  --principal-arn "arn:aws:iam::$(aws sts get-caller-identity --query Account --output text):role/EKSExec"
```

Deleting the access entry cuts off every outstanding EKSExec credential from the cluster immediately.
