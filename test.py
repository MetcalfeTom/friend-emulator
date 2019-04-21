from emulate import train_on_chat
from friend_emulator.data import Message
import pytest

test_data = [
    Message(text=entry)
    for entry in (
        "tom metcalfe : i sure am lonely out here",
        "tom metcalfe : i will keep you company",
    )
]

messages_to_filter = ["tom metcalfe : \n", "tom metcalfe : 👋"]


def test_train_on_chat():
    finished = train_on_chat(test_data)
    assert finished is None
