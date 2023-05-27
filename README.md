# tiny-tree-util

a very minimal, no-dependency implementation of jax's more commonly used tree utilities: tree_flatten, tree_map, tree_unflatten, tree_leaves, tree_reduce. these functions behave the same more or less as jax's implementations, and the API is similar. however this version only supports lists, tuples, dicts and OrderedDicts (as well as arbitrary-type leaf nodes).
To keep things minimal, it doesn't use the same PyTreeDef as jax, and so mixing between them, e.g ``jax.tree_util.tree_unflatten(*tree_flatten(tree))`` wouldnt work. 
