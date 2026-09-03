"""Model wrappers. Each one is a `Protocol` plus one real implementation, so
the pipeline can be built against scripted stand-ins without importing a
runtime. Nothing here is imported eagerly by `vision/__init__.py`.
"""
