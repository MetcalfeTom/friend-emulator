from emulate import train_on_chat

test_data = ["tom metcalfe : i sure am lonely out here",
             "tom metcalfe : i will keep you company"]


def test_train_on_chat():
    finished = train_on_chat(test_data)
    assert finished == 1
