__all__ = ["Config", "selector_matches"]

from typing import Any

from . import KEY_BLOCK_TEMPLATE, KEY_LAYER_CLASS, selector_matches

class ConfigRule:

    specificity = 1

    def __init__(self, selector, value, inheritable=False):
        self.selector = selector
        self.value = value
        self.inheritable = inheritable

    def lineage(self):
        return self.selector[:-1]

    def key(self):
        return self.selector[-1]
    
    def is_more_specific_than(self, other):
        if other is None:
            return True
        
        if self.specificity != other.specificity:
            return self.specificity > other.specificity

        if len(self.lineage()) != len(other.lineage()):
            # deeper is more specific
            return len(self.lineage()) > len(other.lineage())

        # Otherwise, last added selector is more specific
        return True
    
    def matches(self, lineage):
        if self.inheritable:
            return selector_matches(lineage, self.lineage())
        else:
            return lineage == self.lineage()
    

    def __repr__(self):
        return ".".join(self.selector) + " = " + str(self.value)


class ConfigRuleDefault(ConfigRule):
    specificity = 0
    pass



class ConfigRuleWrapper(ConfigRule):
    def __init__(self, selector, value):
        self._selector = selector
        self._child = value
        self.specificity = value.specificity
    
    def lineage(self):
        return self._selector + self._child.lineage()
    
    def key(self):
        return self._child.key()

    @property
    def inheritable(self):
        return self._child.inheritable

    @property
    def value(self):
        return self._child.value
    
    def __repr__(self):
        return ".".join(self.lineage()) + " = " + str(self.value)


class Config:
    def __init__(self, rules=[], selector=()):

        self._rules = rules.copy()
        self._selector = selector


    def __getattr__(self, name):
        return Config(self._rules, self._selector + (name,))
    
    def __call__(self, value, inheritable=False):
        self._rules.append(ConfigRule(self._selector, value, inheritable=inheritable))

        # When you call a config, it should reset the selector
        # This is what allows the chained syntax
        # Config().a.b.c(1).d.e.f(2)
        return Config(self._rules, ())

    
    def set(self, selectors, value, inheritable=False):
        if isinstance(selectors, str):
            selectors = tuple(selectors.split("."))
        self._rules.append(ConfigRule(self._selector + selectors, value, inheritable=inheritable))
        return self
    
    def default(self, selectors, value, inheritable=False):

        if isinstance(selectors, str):
            selectors = tuple(selectors.split("."))

        self._rules.append(ConfigRuleDefault(self._selector + selectors, value, inheritable=inheritable))
        return self
    
    def merge(self, key, config):

        for rule in config._rules:
            self._rules.append(ConfigRuleWrapper(self._selector + (key,), rule))
        return self
    
    def get(self, key, default=None):
        if isinstance(key, str):
            key = tuple(key.split("."))
        lineage = key[:-1]
        key = key[-1]

        if lineage:
            return self.with_selector(lineage).get(key, default)
        else:
            return self.get_parameters().get(key, default)
        # return self.get_parameters().get(key, default)
    

    def with_selector(self, selector):

        if not isinstance(selector, tuple):
            selector = tuple(selector.split("."))

        return Config(self._rules, self._selector + selector)
    
    def get_module(self):
        return self.get_parameters().get(KEY_LAYER_CLASS, None)
    
    def get_parameters(self):
        key_rule_dict = {}

        # Higher depth means more specific
        for rule in self._rules:
            
            if not rule.matches(self._selector):
                continue
            
            key = rule.key()
            
            if not rule.is_more_specific_than(key_rule_dict.get(key, None)):
                continue

            key_rule_dict[key] = rule
            
        output_dict = {key: rule.value for key, rule in key_rule_dict.items()}

        return output_dict
