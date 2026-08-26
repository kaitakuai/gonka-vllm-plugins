# Frozen sphere codebook

The consensus codebook is `sphere_codebook.pt`, sha256
`73d57b52b284c0efb0aaabe47707fd5d281d70abbd58fd04a6c5766a74a3c4b9`.

It ships **inside the plugin package** (`gonka_poc/poc/sphere_codebook.pt`) and
is loaded from there. The seams patch does not carry it and does not need to:
nothing in the engine tree reads a codebook.

If the file is missing, `get_sphere_codebook()` rebuilds one with Adam and warns
`NOT consensus-safe`. That rebuild is not reproducible across machines, so a
node in that state produces artifacts no validator will confirm. Treat the
warning as a hard failure.
