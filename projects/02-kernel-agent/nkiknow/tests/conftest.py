import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))   # projects/02-kernel-agent
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
os.chdir(ROOT)   # agent/nkibench read reference_level*.py relative to cwd
