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
            referenced_value = new_config.get(
                self.value.selectors, return_dict_if_multiple=False
            )
            # Evaluate the ref function
            return self.value(referenced_value)

        if isinstance(self.value, ForwardHook):
            return self.value.value()

        return self.value

    def __repr__(self):
        return (
            str(self.selector)
            + "."
            + str(self.key)
            + " = "
            + str(self.value)
            + (" (default)" if self.default else "")
        )


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

    def __attr__(self, name):
        # defer to the wrapped rule if the attribute is not found
        return getattr(self.value, name)


class ForwardHook:
    def __init__(self, hook, first_only=False):
        if isinstance(hook, ForwardHook):
            hook = hook.hook
            first_only = hook.first_only

        self.hook = hook
        self.first_only = first_only

        self._value = None
        self._has_run = False

    def __call__(self, x):
        if self.first_only and self._has_run:
            return self._value
        self.value = self.hook(x)
        self._has_run = True

    def value(self):
        if not self._has_run:
            raise ValueError(
                "Hook has not been run yet. Make sure the module is evaluated before the target is called."
            )
        return self._value

    def has_run(self):
        return self._has_run


class Config:
    def __init__(self, rules=None, refs=None, context=NoneSelector()):
        self._rules = [] if rules is None else rules.copy()
        self._refs = {} if refs is None else refs
        self._context = context

    def on_first_forward(self, target, hook):
        return self.on_forward(target, hook, first_only=True)

    def on_forward(self, target, hook, first_only=False):
        if not first_only:
            raise NotImplementedError("Only first_only is supported for now")

        target = parse_selectors(target)
        _key = str(target)

        # Create a Ref from target to the current context + _key.
        # On forward, context + _key will be evaluated.
        # The remote rule will automatically reflect the changes.
        self._rules.append(ConfigRule(target, _key, Ref(self._context + _key)))

        # Create a rule that will be evaluated on forward
        self._rules.append(
            ConfigRule(self._context, _key, ForwardHook(hook, first_only=first_only))
        )

        return Config(self._rules, self._refs)

    def run_all_forward_hooks(self, x):
        for rule in self.get_all_forward_hooks():
            rule.value(x)

    def has_forward_hooks(self):
        return len(self.get_all_forward_hooks()) > 0

    def get_all_forward_hooks(self):
        rules = self._get_all_matching_rules(NoneSelector(), match_key=False)
        module = self._get_all_matching_rules(NoneSelector(), match_key=True)

        all_rules = rules + module
        return [rule for rule in all_rules if isinstance(rule.value, ForwardHook)]

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
                        (body + new_head.selector)[i] + selectors, str(key), value
                    )
                )

        else:
            if length is None and not hasattr(generator, "__len__"):
                # Warn that we will only populate up to index 256
                # Only warn once
                warnings.warn(
                    """Populating a config with a generator of unknown length without specifying the length will only populate up to index 256.
To populate more, specify the length with .populate(..., length=desired_length)"""
                )

            length = length or 256

            if callable(generator):
                generator = [generator(i) for i in range(length)]

            # Is there any case where this is not equivalent to enumerate?
            for i, value in zip(range(length), generator):
                self._rules.append(
                    ConfigRule(self._context[i] + selectors, str(key), value)
                )

        return Config(self._rules, self._refs)

    def default(self, selectors, value):
        return self.set(selectors, value, default=True)

    def merge(self, selectors, config, as_default=False, prepend=False):
        selectors = parse_selectors(selectors)

        additional_rules = []
        for rule in config._rules:
            wrapped_rule = ConfigRuleWrapper(
                self._context + selectors, rule, default=as_default
            )
            additional_rules.append(wrapped_rule)

        if prepend:
            self._rules = additional_rules + self._rules
        else:
            self._rules = self._rules + additional_rules

        return Config(self._rules, self._refs)

    def get(self, selectors, default=None, return_dict_if_multiple=False):
        selectors = parse_selectors(selectors)
        full_context = self._context + selectors

        # check if the last selector is a index selector
        last_selector_is_index = isinstance(full_context.pop()[-1], IndexSelector)

        rules = self._get_all_matching_rules(
            selectors, match_key=True, allow_indexed=True
        )
        rules_per_key = self._merge_rules_on_key(rules)

        if not last_selector_is_index:
            most_specific = self._take_most_specific_per_key_and_index(
                rules_per_key, self
            )
        else:
            most_specific = self._take_most_specific_per_key(rules_per_key, self)

        if len(most_specific) == 0:
            return default
        if len(most_specific) == 1:
            return list(most_specific.values())[0]
        if return_dict_if_multiple:
            return most_specific

        raise ValueError(
            f"Multiple keys match {selectors} ({list(most_specific.keys())})"
        )

    def get_module(self):
        return self.get(NoneSelector())

    def add_ref(self, name, config):
        if name in self._refs:
            warnings.warn(
                f"UID {name} already exists with value {self._refs[name]}. It will be overwritten."
            )
        self._refs[name] = config

    def get_ref(self, name):
        return self._refs[name]

    def get_parameters(self):
        rules = self._get_all_matching_rules(NoneSelector(), match_key=False)
        rule_dict = self._merge_rules_on_key(rules)
        return self._take_most_specific_per_key(rule_dict, self)

    def with_selector(self, selectors):
        selectors = parse_selectors(selectors)
        return Config(self._rules, self._refs, self._context + selectors)

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
            raise ValueError(
                "Cannot index a config with no context. Use a class selector first"
            )

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

    def __repr__(self):
        return "Config(\n" + "\n".join([str(rule) for rule in self._rules]) + "\n)"

    def _get_all_matching_rules(self, selectors, match_key=True, allow_indexed=False):
        contextualized_selectors = self._context + selectors
        return [
            rule
            for rule in self._rules
            if rule.matches(
                contextualized_selectors,
                match_key=match_key,
                allow_indexed=allow_indexed,
            )
        ]

    def _is_last_selector_a(self, type):
        if isinstance(self._context, NoneSelector):
            return type == NoneSelector
        _, head = self._context.pop()
        return isinstance(head, type)

    @staticmethod
    def _take_most_specific_per_key(key_rule_dict, config):
        most_specific = {}
        for key, rules in key_rule_dict.items():
            most_specific[key] = Config._take_most_specific_in_list(rules).get_value(
                config
            )
        return most_specific

    @staticmethod
    def _take_most_specific_in_list(rules):
        most_specific = rules[0]
        for rule in rules:
            if rule.is_more_specific_than(most_specific):
                most_specific = rule
        return most_specific

    @staticmethod
    def _take_most_specific_per_key_and_index(key_rule_dict, config):
        # This function is awful. It needs to be rewritten.

        most_specific = {}

        for key, rules in key_rule_dict.items():
            rule_if_no_indexed = []
            any_indexed = False

            indexed_values = {}
            for rule in rules:
                if isinstance(rule.head, IndexSelector):
                    any_indexed = True
                    for index in rule.head.get_list_of_indices():
                        if index not in indexed_values:
                            indexed_values[index] = [rule]
                        else:
                            indexed_values[index].append(rule)
                else:
                    rule_if_no_indexed.append(rule)

            if len(rule_if_no_indexed) == 0:
                least_specific_rule = ConfigRule(NoneSelector(), "", [])
                least_specific_rule.specificity = -9999
                rule_if_no_indexed = [least_specific_rule]

            most_specific_rule_if_no_index = Config._take_most_specific_in_list(
                rule_if_no_indexed
            )
            value_if_no_indexed = most_specific_rule_if_no_index.get_value(config)

            if not any_indexed:
                most_specific[key] = value_if_no_indexed
                continue

            # We will use this to fill potential missing indices
            if not isinstance(value_if_no_indexed, (list, tuple)):
                value_if_no_indexed = [value_if_no_indexed]

            for idx, value in enumerate(value_if_no_indexed):
                if idx not in indexed_values or (
                    most_specific_rule_if_no_index.specificity
                    > max(r.specificity for r in indexed_values[idx])
                ):
                    indexed_values[idx] = [most_specific_rule_if_no_index]

            most_specific_for_key = {}
            for index, rules in indexed_values.items():
                most_specific_for_key[index] = Config._take_most_specific_in_list(rules)

            indices = list(indexed_values.keys())
            indices.sort()
            missing_indices = [i for i in range(indices[-1]) if i not in indices]
            assert (
                len(missing_indices) == 0
            ), f"Missing indices {missing_indices} for key {key}"

            most_specific_values = [
                most_specific_for_key[i].get_value(config) for i in indices
            ]
            for i in indices:
                if not isinstance(most_specific_for_key[i].head, IndexSelector):
                    most_specific_values[i] = most_specific_values[i][i]
            most_specific[key] = most_specific_values

        return most_specific

    @staticmethod
    def _merge_rules_on_key(rules):
        merged = {}
        for rule in rules:
            key = rule.key
            if key in merged:
                merged[key].append(rule)
            else:
                merged[key] = [rule]
        return merged
