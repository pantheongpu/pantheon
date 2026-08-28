# Graph Replay

Captures one GPU kernel sequence and repeatedly replays the resulting execution graph. CUDA builds use CUDA Graphs through the HIP compatibility API; ROCm builds use HIP Graphs. Run `pantheon --test graph_replay --duration 60 --gpu 0 --mem 25 --verify --profile`.
