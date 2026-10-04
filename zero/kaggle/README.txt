Kaggle reproduces a committed sha; it never decides anything (deciding numbers: local RTX 4060).

1. Commit the runner. Optional: kaggle/private.json (gitignored) = {"user","dataset_slug","kernel_slug"}.
2. python kaggle/build.py --sha <sha> --runner run_t4a.py --runner-args "<args>"
   Refuses uncommitted shas and untracked runners. Writes kaggle/out/ (gitignored).
3. build.py only prints the push commands, prefixed "Seal only:". Run them yourself:
     kaggle datasets create -p kaggle/out/dataset
     kaggle kernels push -p kaggle/out/kernel
4. The kernel clones the bundle, checks out <sha>, asserts HEAD == sha, runs the runner with
   ZERO_OUT=/kaggle/working/out, logs to repro.log and writes REPRO.json (sha, rc, gpu, versions).
Token stays in ~/.kaggle/kaggle.json, local only. Nothing here reads it.
