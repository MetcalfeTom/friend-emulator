from emulate import train_on_chat, pre_process, NoFriendsError

test_data = [
    "tom metcalfe : i sure am lonely out here",
    "tom metcalfe : i will keep you company",
]

messages_to_filter = ["tom metcalfe : \n", "tom metcalfe : 👋"]


def test_train_on_chat():
    finished = train_on_chat(test_data)
    assert finished == 1


def test_pre_process():
    try:
        pre_process(messages_to_filter)
        assert False

    except NoFriendsError:
        assert True
