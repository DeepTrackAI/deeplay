from __future__ import annotations


class A:
    def __init__(self, value):
        self.value = value

    def __getattr__(self, name):
        return A(name)
