#!/usr/bin/env bash
# Stop post-processing jobs (tmux post + their vLLM/evalplus children).
tmux kill-session -t post 2>/dev/null; sleep 2
ps -eo pid,args | awk '/eval\/generate.py|eval\/attempts.py|eval\/run_eval.sh|eval\/run_paraphrase.sh|evalplus.evaluate|train\/fast_post.sh|EngineCore/ && !/awk/ && !/stop_post/ {print $1}' | xargs -r kill 2>/dev/null; sleep 3
ps -eo pid,args | awk '/EngineCore|eval\/generate.py/ && !/awk/ && !/stop_post/ {print $1}' | xargs -r kill -9 2>/dev/null; true
