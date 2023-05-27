try:
    import jax
    import numpy as np
except BaseException as e:
    print(e)
    raise RuntimeError("the test does not work without jax and numpy.")

from tree_util import *

c = 1
none_tree = None
object_tree = np.array([-1, -2, -3])
tuple_tree = (np.array([1, 2, 3]),)
list_tree = [np.array([4, 5, 6])]
simple_tree = {
    "ins": [
        {
            "w1": 1 * np.ones([c, c]),
            "b1": 2 * np.ones([c, c]),
        },
        {
            "w2": 3 * np.ones([c, c]),
            "b2": 4 * np.ones([c, c]),
        },
        [1, 2, 3],
        None
    ],
    "out": 5 * np.ones([c, c]),
}
fancy_tree = {
    "ins": [
        {
            "w1": 1 * np.ones([c, c]),
            "b1": 2 * np.ones([c, c]),
            "nested": OrderedDict({
                "x": np.array([1, 2, 3]),
                "y": {
                    "z1": np.array([4, 5, 6]),
                    "z2": np.array([7, 8, 9])
                }
            })
        },
        {
            "w2": 3 * np.ones([c, c]),
            "b2": 4 * np.ones([c, c]),
            "nested_list": [
                np.array([10, 11, 12]),
                {
                    "a": np.array([13, 14, 15]),
                    "b": np.array([16, 17, 18])
                },
                (
                    np.array([19, 20, 21]),
                    np.array([22, 23, 24])
                )
            ]
        },
        [1, 2, 3],
    ],
    "out": 5 * np.ones([c, c]),
    "nested_dict": {
        "d1": {
            "e1": OrderedDict({
                "f1": np.array([25, 26, 27]),
                "f2": np.array([28, 29, 30])
            }),
            "e2": np.array([31, 32, 33])
        },
        "d2": {
            "e3": np.array([34, 35, 36]),
            "e4": np.array([37, 38, 39])
        },
        "nested_tuple": (
            np.array([40, 41, 42]),
            np.array([43, 44, 45])
        )
    },
    "nested_list": [
        np.array([46, 47, 48]),
        (
            np.array([49, 50, 51]),
            np.array([52, 53, 54])
        ),
        {
            "g1": np.array([55, 56, 57]),
            "g2": np.array([58, 59, 60])
        },
        None,
        [None, None, True]
    ]
}
trees = [none_tree, object_tree, tuple_tree, list_tree, simple_tree, fancy_tree]

def is_same(a, b):
    if isinstance(a, np.ndarray):
        if not isinstance(b, np.ndarray):
            return False
        return np.allclose(a, b)
    else:
        return a == b

def test_tree(test_name, tree, correct_tree, test_tree):
    try:
        comparison = jax.tree_util.tree_map(lambda a, b: is_same(a, b), correct_tree, test_tree)
        assert all(jax.tree_util.tree_flatten(comparison)[0])
    except:
        raise RuntimeError(f"failed {test_name} test for tree {tree}. correct tree is {correct_tree}, but got {test_tree}")
    return True

for tree in trees:
    correct_flat_tree, correct_pytree_def = jax.tree_util.tree_flatten(tree)
    test_flat_tree, test_pytree_def = tree_flatten(tree)
    test_tree('tree_flatten', tree, correct_flat_tree, test_flat_tree)

    correct_unflat_tree = jax.tree_util.tree_unflatten(correct_pytree_def, correct_flat_tree)
    test_unflat_tree = tree_unflatten(test_pytree_def, test_flat_tree)
    test_tree('tree_unflatten', tree, correct_unflat_tree, test_unflat_tree)
    
    fun1 = lambda x: x * 2
    fun2 = lambda x, y: x - y
    fun3 = lambda x: type(x)

    correct_mapped_tree = jax.tree_util.tree_map(fun1, tree)
    test_mapped_tree = tree_map(fun1, tree)
    test_tree(f'tree_map #1', tree, correct_mapped_tree, test_mapped_tree)
    
    correct_2arg_tree = jax.tree_util.tree_map(fun2, tree, correct_mapped_tree)
    test_2arg_tree = jax.tree_util.tree_map(fun2, tree, test_mapped_tree)
    test_tree(f'2-input tree_map', tree, correct_flat_tree, test_flat_tree)

    correct_mapped_tree = jax.tree_util.tree_map(fun3, tree)
    test_mapped_tree = tree_map(fun3, tree)
    test_tree(f'tree_map #3', tree, correct_mapped_tree, test_mapped_tree)
