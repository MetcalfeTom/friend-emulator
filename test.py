from emulate import train_on_chat
from friend_emulator.data import Message

test_data = [
    Message(user="thomas metcalfe", text="i sure am lonely out here"),
    Message(user="thomas metcalfe", text="i will keep you company"),
]

messages_to_filter = ["tom metcalfe : \n", "tom metcalfe : 👋"]


def test_train_on_chat(capsys):
    train_on_chat(test_data)
    captured = capsys.readouterr()[0]

    assert "loss:" in captured
    assert "thomas" in captured
    assert "------------------------------------------------" in captured