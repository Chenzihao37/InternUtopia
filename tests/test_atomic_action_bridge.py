from internutopia.bridge.atomic_actions import parse_coherent_action


def test_parse_movetowards_action():
    action = parse_coherent_action("<robot dog>(23): [movetowards] <door>(9)")

    assert action.agent_name == "robot dog"
    assert action.agent_id == 23
    assert action.action == "movetowards"
    assert action.object_name == "door"
    assert action.object_id == 9
    assert action.bridge_action == "navigate"


def test_parse_puton_action():
    action = parse_coherent_action("<robot arm>(24): [puton] <milkbox>(30) on <dining table>(13)")

    assert action.action == "puton"
    assert action.relation == "on"
    assert action.target_name == "dining table"
    assert action.target_id == 13
    assert action.bridge_action == "place"


def test_parse_putinto_action():
    action = parse_coherent_action("<robot arm>(24): [putinto] <milkbox>(30) into <plate>(51)")

    assert action.action == "putinto"
    assert action.relation == "into"
    assert action.target_name == "plate"
    assert action.target_id == 51
