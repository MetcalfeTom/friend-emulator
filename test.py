from emulate import train_on_chat
from friend_emulator.data import Message

test_data = [
    Message(user="thomas metcalfe", text="i sure am lonely out here"),
    Message(user="thomas metcalfe", text="i will keep you company"),
]

messages_to_filter = ["tom metcalfe : \n", "tom metcalfe : 👋"]


def test_train_on_chat(capsys):
    finished = train_on_chat(test_data)
    captured = capsys.readouterr()
    # assert "loss:" in captured.out
    # assert "tom" in captured.out
    # assert "------------------------------------------------" in captured.out
    assert finished is None
