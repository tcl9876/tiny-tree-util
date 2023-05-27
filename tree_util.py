import functools
from collections import OrderedDict

class LeafType:
    def __init__(self, is_none=False):
        self.is_none = is_none
    
    def __repr__(self):
        return "*" if self.is_none else "None"

def tree_flatten(tree, leaf_predicate=None):
    leaves = []

    def flatten_impl(handle, leaves, leaf_predicate=None):
        treedef = LeafType()
        
        if leaf_predicate and leaf_predicate(handle):
            leaves.append(handle)
        else:
            def recurse(child):
                return flatten_impl(child, leaves, leaf_predicate)
            
            if handle is None:
                treedef.is_none = True
            elif isinstance(handle, (tuple, list)):
                treedef = []
                for i in range(len(handle)):
                    treedef.append(recurse(handle[i]))
                treedef = type(handle)(treedef)
            elif isinstance(handle, dict):
                treedef = {}
                for key in sorted(handle.keys()):
                    treedef[key] = recurse(handle[key])
                if isinstance(handle, OrderedDict):
                    treedef = OrderedDict(treedef)
            else:
                leaves.append(handle)

        return treedef

    return (leaves, flatten_impl(tree, leaves, leaf_predicate))

def tree_unflatten(treedef, tree_leaves):
    tree_leaves = tree_leaves[::-1]
    
    def unflatten_impl(handle, leaves):
        is_none = isinstance(handle, LeafType) and handle.is_none
        treedef = LeafType(is_none)
        
        def recurse(child):
            return unflatten_impl(child, leaves)
        
        if is_none:
            pass
        elif isinstance(handle, (tuple, list)):
            treedef = []
            for i in range(len(handle)):
                treedef.append(recurse(handle[i]))
            treedef = type(handle)(treedef)
        elif isinstance(handle, dict):
            treedef = {}
            for key in sorted(handle.keys()):
                treedef[key] = recurse(handle[key])
            if isinstance(handle, OrderedDict):
                treedef = OrderedDict(treedef)
        else:
            if len(tree_leaves) == 0:
                raise RuntimeError("there are more tree_leaves in argument #0 than specified by the treedef in argument #1")
            treedef = tree_leaves.pop()

        return treedef
    
    unflattened_tree = unflatten_impl(treedef, tree_leaves)
    if len(tree_leaves) > 0:
        raise RuntimeError("there are fewer tree_leaves in argument #0 than specified by the treedef in argument #1")
    return unflattened_tree

def tree_leaves(tree):
    return tree_flatten(tree)[0]

def tree_structure(tree):
    return tree_flatten(tree)[1]

def tree_map(f, tree, *rest):
    all_flattened_trees = [tree_flatten(t) for t in (tree, ) + rest]
    all_leaves = [r[0] for r in all_flattened_trees]
    all_treedefs = [r[1] for r in all_flattened_trees]
    for i, treedef in enumerate(all_treedefs):
        if repr(treedef) != repr(all_treedefs[0]):
            raise RuntimeError(f"Got a tree with a different structure: tree #0 has structure \n{repr(all_treedefs[0])}\n but tree #{i} has structure \n{repr(treedef)}")
        
    return tree_unflatten(all_treedefs[0], [f(*xs) for xs in zip(*all_leaves)])

def tree_reduce(function, tree):
    return functools.reduce(function, tree_leaves(tree))