import os
import io


class Message(dict):
    """WhatsApp message as a dictionary."""

    def __init__(self, user=None, text=None, timestamp=None):
        keys = ["user", "text", "timestamp"]
        values = [user, text, timestamp]
        mapping = zip(keys, values)

        super(Message, self).__init__(mapping)


class DataFriend(object):
    """That kid you went to school with who loads and processes datasets."""

    def __init__(self, log_directory=os.getcwd()):
        self.direc = log_directory
        self.messages = []

        self.load_chat()
        self.pre_process()

    def load_chat(self):
        """Loads WhatsApp chat logs from given directory."""

        files = [
            os.path.join(self.direc, f)
            for f in os.listdir(self.direc)
            if "WhatsApp Chat" in f
        ]

        # append all chats to one list
        for file in files:
            with io.open(file, encoding="utf-8") as TextFile:
                self.messages.extend(TextFile.readlines())

        if len(self.messages) == 0:
            raise NoFriendsError

    ##TODO: keep timestamps and users
    # TODO: refactor for quicker processing
    def pre_process(self):
        """Takes a list of messages and removes all timestamps & media messages.
           Returns a list of messages with all special characters and emoji removed"""

        # remove media message remnants & timestamps

        nomedia = [
            message[20:].lower()
            for message in self.messages
            if "<Media omitted>" not in message
        ]

        m = len(nomedia)

        filter = list(
            " eaotinslrhdmcy\nu:gwbkfpj'xv?.054,7912/\"3!z-q86+£=(_)&*%@#;$\\~^<[]>"
        )

        processed = []

        for i in range(m):
            filtered = ""
            for char in nomedia[i]:
                if char in filter:
                    filtered += char

            # omit completely filtered (blank) messages
            if ": \n" not in filtered and len(filtered) > 1:
                msg = Message(text=filtered.strip(" -"))

                processed.append(msg)

        if len(processed) == 0:
            raise NoFriendsError

        self.messages = processed


class NoFriendsError(Exception):
    """Raised when the message log is empty after filtering."""

    def __init__(self, message=None):
        self.message = message or "Message list is empty."

    def __str__(self):
        return self.message
