from rebuild_newton_contact_layer import contiguous_runs, manifold_runs, semantic_body


def test_shape_names_keep_heel_and_toe_distinct():
    assert semantic_body('/Robot/left_ankle_roll_sphere_1_link/sphere','left_foot') == 'left_heel'
    assert semantic_body('/Robot/left_ankle_roll_sphere_5','left_foot') == 'left_toe'
    assert semantic_body('/Robot/left_ankle_roll_link/mesh','left_foot') == 'left_foot'


def test_contacts_are_not_closed_across_gaps():
    assert contiguous_runs([0,1,4,5]) == [[0,2],[4,6]]


def test_manifold_count_changes_preserve_all_samples():
    frames={0:[1],1:[1],2:[1,2],3:[1,2],5:[1]}
    runs=manifold_runs(frames)
    assert runs == [(0,2),(2,4),(5,6)]
    assert sum((b-a)*len(frames[a]) for a,b in runs) == sum(map(len,frames.values()))
