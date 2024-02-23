from dataclasses import dataclass, asdict


@dataclass
class Prompt:
    id: int
    prompt: str

    def to_dict(self):
        '''Converts the dataclass to a dictionary for serialization'''
        return asdict(self)


@dataclass
class Context:
    setting: str
    tone: str
    conversation_type: str

    def to_dict(self):
        '''Converts the dataclass to a dictionary for serialization'''
        return asdict(self)


@dataclass
class Bio:
    name: str
    age: int
    about_me: str

    def to_dict(self):
        '''Converts the dataclass to a dictionary for serialization'''
        return asdict(self)


@dataclass
class TestCase:
    id: int
    utterance: str
    context: Context
    bio: Bio
    good_completions: list[str]

    def to_dict(self):
        '''Converts the dataclass to a dictionary for serialization'''
        return asdict(self)
