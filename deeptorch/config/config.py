__all__ = ["Config", "Ref"]

import re
import warnings

from . import (
    KEY_BLOCK_TEMPLATE, 
    KEY_LAYER_CLASS, 
    ClassSelector, 
    WildcardSelector,
    DoubleWildcardSelector,
    IndexSelector,
    NoneSelector,
    Ref,
    parse_selectors,
)

class ConfigRule:

    specificity = 1
    default = False

    def __init__(self, selector, key, value):
        self.selector = selector
        
        head = parse_selectors(key)

        self.head = head
        self.key = head.key()
        self.value = value
        self.scope_root = NoneSelector()

        # self._selector_has_wildcard = selector.any(lambda s: isinstance(s, WildcardSelector))
        # self._selector_has_double_wildcard = selector.any(lambda s: isinstance(s, DoubleWildcardSelector))
        
    
    def is_more_specific_than(self, other):
        if other is None:
            return True
        
        if self.specificity != other.specificity:
            return self.specificity > other.specificity

        # Otherwise, last added selector is more specific
        # In the future, we should consider wildcard selectors as less specific
        return True
    
    def matches(self, context, match_key=False, allow_indexed=False):
        
        head = ClassSelector(self.key) if allow_indexed else self.head
        if match_key:
            full_selector = self.selector + head
        else:
            full_selector = self.selector

        
        # Handle the case where either or both the full selector and the context are None
        full_selector_is_none = isinstance(full_selector, NoneSelector)
        context_is_none = isinstance(context, NoneSelector)
        if full_selector_is_none and context_is_none:
            return True
        if full_selector_is_none or context_is_none:
            return False

        regex = full_selector.regex()



        for selector in context:
            if re.match(regex, selector):
                return True
            
        return False
    
    def get_value(self, config):

        if isinstance(self.value, Ref):
            # Create a new config with the root context of the rule
            new_config = Config(config._rules, config._refs, self.scope_root)
            # Get the value of the ref, should be a unique selector
            referenced_value = new_config.get(self.value.selectors, return_dict_if_multiple=False)
            # Evaluate the ref function
            return self.value(referenced_value)
            
        return self.value

    def __repr__(self):
        return str(self.selector) + "." + str(self.key) + " = " + str(self.value) + (" (default)" if self.default else "")


class ConfigRuleDefault(ConfigRule):
    specificity = 0
    default = True


class ConfigRuleWrapper(ConfigRule):
    def __init__(self, selector, value, default=False):

        if default:
            self.specificity = 0
            self.default = True

        self.selector = selector + value.selector
        self.scope_root = selector + value.scope_root
        self.key = value.key
        self.head = value.head
        self.value = value.value


class Config:
    
    
    def __init__(self, rules=[], refs=None, context=NoneSelector()):

        self._rules = rules.copy()
        self._refs = {} if refs is None else refs.copy()
        self._context = context

    def __getattr__(self, name):
        match name:
            case "_":
                selector = WildcardSelector()
            case "__":
                selector = DoubleWildcardSelector()
            case _:
                selector = ClassSelector(name)
        return Config(self._rules, self._refs, self._context + selector)
    
    def __getitem__(self, index):
        if isinstance(self._context, NoneSelector):
            raise ValueError("Cannot index a config with no context. Use a class selector first")

        if isinstance(index, tuple):
            index, length = index
        else:
            length = None
        return Config(self._rules, self._refs, self._context[index, length])
    
    def __call__(self, *x, **kwargs):
        
        if len(x) > 1:
            raise ValueError("Config can only be called with one positional argument")
        
        if len(x) == 1:
            x = x[0]
            selector, key = self._context.pop()

            self._rules.append(ConfigRule(selector, key, x))

        for key, value in kwargs.items():
            rule = ConfigRule(self._context, key, value)
            self._rules.append(rule)

        # When you call a config, it should reset the selector
        # This is what allows the chained syntax
        # Config().a.b.c(1).d.e.f(2)
        return Config(self._rules, self._refs)
    
    

    def set(self, selectors, value, default=False):
        selectors = parse_selectors(selectors)

        selectors, key = selectors.pop()
        if default:
            self._rules.append(ConfigRuleDefault(self._context + selectors, key, value))
        else:
            self._rules.append(ConfigRule(self._context + selectors, key, value))
        return self
    
    def populate(self, selectors, generator, length=None):
        # idx is interpreted as follows:
        # None: every index.
        # int: all indices up to but not including idx

        selectors = parse_selectors(selectors)
        selectors, key = selectors.pop()

        body, head = self._context.pop()


        if isinstance(head, IndexSelector):
            new_head = IndexSelector(head.selector, head.index, length)
            integers_to_populate = new_head.get_list_of_indices()
            
            if callable(generator):
                generator = [generator(i) for i in integers_to_populate]

            for i, value in zip(integers_to_populate, generator):
                self._rules.append(
                    ConfigRule(
                        (body + new_head.selector)[i] + selectors,
                        str(key),
                        value
                    )
                )

        else:
            if length is None and not hasattr(generator, "__len__"):
                # Warn that we will only populate up to index 256
                # Only warn once
                warnings.warn(
"""Populating a config with a generator of unknown length without specifying the length will only populate up to index 256.
To populate more, specify the length with .populate(..., length=desired_length)""")

            length = length or 256

            if callable(generator):
                generator = [generator(i) for i in range(length)]

            # Is there any case where this is not equivalent to enumerate?
            for i, value in zip(range(length), generator):
                self._rules.append(
                    ConfigRule(
                        self._context[i] + selectors,
                        str(key),
                        value
                    )
                )
        
        return Config(self._rules, self._refs)

    def default(self, selectors, value):
        return self.set(selectors, value, default=True)
    
    def merge(self, selectors, config, as_default=False, prepend=False):
        
        selectors = parse_selectors(selectors)

        additional_rules = []
        for rule in config._rules:
            wrapped_rule = ConfigRuleWrapper(self._context + selectors, rule, default=as_default)
            additional_rules.append(wrapped_rule)

        if prepend:
            self._rules = additional_rules + self._rules
        else:
            self._rules = self._rules + additional_rules

        return Config(self._rules, self._refs)
    
    def get(self, selectors, default=None, return_dict_if_multiple=False):
        
        selectors = parse_selectors(selectors)
        rules = self._get_all_matching_rules(selectors, match_key=True)
        most_specific = self._take_most_specific_per_key(rules)

        if len(most_specific) == 0:
            return default
        if len(most_specific) == 1:
            return list(most_specific.values())[0].get_value(self)
        if return_dict_if_multiple:
            return {key: rule.get_value(self) for key, rule in most_specific.items()}
        
        raise ValueError(f"Multiple keys match {selectors} ({list(most_specific.keys())})")

    def get_module(self):
        return self.get(NoneSelector())
    
    def add_ref(self, name, config):
        if name in self._refs:
            warnings.warn(f"UID {name} already exists with value {self._refs[name]}. It will be overwritten.")
        self._refs[name] = config

    def get_ref(self, name):
        return self._refs[name]

    def get_parameters(self):
        rules = self._get_all_matching_rules(NoneSelector(), match_key=False)
        most_specific = self._take_most_specific_per_key(rules)
        return {key: rule.get_value(self) for key, rule in most_specific.items()}

    def with_selector(self, selectors):
        selectors = parse_selectors(selectors)
        return Config(self._rules, self._refs, self._context + selectors)
    
    def __repr__(self):
        return "Config(\n" + "\n".join([str(rule) for rule in self._rules]) + "\n)"

    def _get_all_matching_rules(self, selectors, match_key=True, allow_indexed=False):
        contextualized_selectors = self._context + selectors
        return [
            rule for rule in self._rules 
                 if rule.matches(contextualized_selectors, match_key=match_key, allow_indexed=allow_indexed)
        ]
    
    def _is_last_selector_a(self, type):
        if isinstance(self._context, NoneSelector):
            return type == NoneSelector
        body, head = self._context.pop()
        return isinstance(head, type)

    @staticmethod
    def _take_most_specific_per_key(rules):
        most_specific = {}
        for rule in rules:
            key = rule.key
            if key in most_specific:
                if rule.is_more_specific_than(most_specific[key]):
                    most_specific[key] = rule
            else:
                most_specific[key] = rule
        return most_specific

    