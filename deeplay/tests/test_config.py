import unittest
import re
from ..core.config import (
    Config,
    ClassSelector,
    IndexSelector,
    WildcardSelector,
    DoubleWildcardSelector,
    Ref,
)

from ..core.selectors import ParentalRelation


class TestConfig(unittest.TestCase):
    # def test_selector_matches_1(self):
    #     self.assertTrue(selector_matches(("foo", "bar"), ("foo", "bar")))

    # def test_selector_matches_2(self):
    #     self.assertTrue(selector_matches(("foo", "bar"), ("bar",)))

    # def test_selector_matches_3(self):
    #     self.assertFalse(selector_matches(("foo", "bar"), ("baz",)))

    # def test_selector_matches_4(self):
    #     self.assertFalse(selector_matches(("foo", "bar"), ("foo", "baz")))

    # def test_selector_matches_5(self):
    #     self.assertFalse(selector_matches(("foo", "bar"), ("foo", "bar", "baz")))

    # def test_selector_matches_6(self):
    #     self.assertTrue(selector_matches(("foo", "bar", "baz"), ("foo", "baz")))

    # def test_selector_matches_6(self):
    #     self.assertFalse(selector_matches(("foo", "bar", "baz"), ("baz", "foo")))

    # def test_selector_matches_7(self):
    #     self.assertFalse(selector_matches((), ("foo")))

    # def test_selector_matches_8(self):
    #     self.assertTrue(selector_matches((), ()))

    # def test_selector_matches_9(self):
    #     self.assertTrue(selector_matches(("foo",), ()))

    # def test_selector_matches_10(self):
    #     self.assertTrue(selector_matches(("foo", "bar", "foo", "bar"), ("foo", "bar", "foo", "bar")))

    def test_ClassSelector_to_string(self):
        selector = ClassSelector("foo")
        self.assertEqual(str(selector), "foo")

        selector = ClassSelector("foo bar")
        self.assertEqual(str(selector), "(foo bar)")

    def test_ClassSelector_iter(self):
        selector = ClassSelector("foo bar")
        self.assertEqual(list(selector), ["foo", "bar"])

    def test_WildcardSelector_to_string(self):
        selector = WildcardSelector()
        self.assertEqual(str(selector), "*")

    def test_WildcardSelector_iter(self):
        selector = WildcardSelector()
        with self.assertRaises(NotImplementedError):
            list(selector)

    def test_DoubleWildcardSelector_to_string(self):
        selector = DoubleWildcardSelector()
        self.assertEqual(str(selector), "**")

    def test_DoubleWildcardSelector_iter(self):
        selector = DoubleWildcardSelector()
        with self.assertRaises(NotImplementedError):
            list(selector)

    def test_IndexSelector_to_string(self):
        parent = ClassSelector("foo")
        selector = IndexSelector(parent, 0)
        self.assertEqual(str(selector), "foo[0]")

        selector = IndexSelector(parent, slice(0, 2))
        self.assertEqual(str(selector), "foo[slice(0, 2, None)]")

    def test_IndexSelector_iter(self):
        parent = ClassSelector("foo")
        selector = IndexSelector(parent, 0)
        self.assertEqual(list(selector), ["foo[0]"])

        selector = IndexSelector(parent, slice(0, 2))
        self.assertEqual(list(selector), ["foo[0]", "foo[1]"])

        with self.assertRaises(TypeError):
            list(IndexSelector(parent, slice(0, None, 2)))

    def test_ParentalRelation_to_string(self):
        parent = ClassSelector("foo")
        child = ClassSelector("bar")
        selector = ParentalRelation(parent, child)
        self.assertEqual(str(selector), "foo.bar")

    def test_ParentalRelation_iter(self):
        parent = ClassSelector("foo bar")
        child = ClassSelector("baz bix")
        selector = ParentalRelation(parent, child)
        self.assertEqual(list(selector), ["foo.baz", "foo.bix", "bar.baz", "bar.bix"])

    def test_selector_regex_1(self):
        rule = ClassSelector("foo") + ClassSelector("bar")
        tests = [
            ("foo.bar", True),
            ("foo", False),
            ("bar", False),
            ("baz", False),
            ("foo.baz", False),
            ("foo.baz.bar", False),
        ]
        for test, expected in tests:
            regex = rule.regex()
            self.assertEqual(re.match(regex, test) is not None, expected)

    def test_selector_regex_2(self):
        rule = ClassSelector("foo") + WildcardSelector() + ClassSelector("bar")
        tests = [
            ("foo.bar", False),
            ("foo", False),
            ("bar", False),
            ("baz", False),
            ("foo.baz.bar", True),
            ("foo.baz.bar.bix", False),
        ]
        for test, expected in tests:
            regex = rule.regex()
            self.assertEqual(
                re.match(regex, test) is not None,
                expected,
                msg=f"Test: {test} with regex {regex}",
            )

    def test_selector_regex_3(self):
        rule = WildcardSelector() + ClassSelector("bar") + WildcardSelector()
        tests = [
            ("foo.bar", False),
            ("foo", False),
            ("bar", False),
            ("baz", False),
            ("foo.bar.bar", True),
            ("foo.baz.bar.bix", False),
        ]
        for test, expected in tests:
            regex = rule.regex()
            self.assertEqual(
                re.match(regex, test) is not None,
                expected,
                msg=f"Test: {test} with regex {regex}",
            )

    def test_selector_regex_4(self):
        rule = WildcardSelector() + WildcardSelector() + ClassSelector("bar")
        tests = [
            ("foo.bar", False),
            ("foo", False),
            ("bar", False),
            ("baz", False),
            ("foo.bar.bar", True),
            ("foo.baz.bar.bix", False),
        ]
        for test, expected in tests:
            regex = rule.regex()
            self.assertEqual(
                re.match(regex, test) is not None,
                expected,
                msg=f"Test: {test} with regex {regex}",
            )

    def test_selector_regex_5(self):
        rule = ClassSelector("foo") + DoubleWildcardSelector() + ClassSelector("bar")
        tests = [
            ("foo.bar", True),
            ("foo", False),
            ("bar", False),
            ("baz", False),
            ("foo.bar.bar", True),
            ("foo.baz.bar", True),
            ("foo.baz.bar.bix", False),
        ]
        for test, expected in tests:
            regex = rule.regex()
            self.assertEqual(
                re.match(regex, test) is not None,
                expected,
                msg=f"Test: {test} with regex {regex}",
            )

    def test_selector_regex_6(self):
        rule = DoubleWildcardSelector() + ClassSelector("bar")
        tests = [
            ("foo.bar", True),
            ("foo", False),
            ("bar", True),
            ("baz", False),
            ("foo.bar.bar", True),
            ("foo.baz.bar", True),
            ("foo.baz.bar.bix", False),
        ]
        for test, expected in tests:
            regex = rule.regex()
            self.assertEqual(
                re.match(regex, test) is not None,
                expected,
                msg=f"Test: {test} with regex {regex}",
            )

    def test_selector_regex_7(self):
        rule = ClassSelector("foo") + DoubleWildcardSelector()
        tests = [
            ("foo.bar", True),
            ("foo", True),
            ("bar", False),
            ("baz", False),
            ("foo.bar.bar", True),
            ("foo.baz.bar", True),
            ("foo.baz.bar.bix", True),
        ]
        for test, expected in tests:
            regex = rule.regex()
            self.assertEqual(
                re.match(regex, test) is not None,
                expected,
                msg=f"Test: {test} with regex {regex}",
            )

    def test_selector_regex_8(self):
        rule = IndexSelector(ClassSelector("foo"), 0) + ClassSelector("bar")
        tests = [
            ("foo[0].bar", True),
            ("foo[0].bar[0]", True),
            ("foo[1].bar", False),
            ("foo[0]", False),
            ("bar", False),
            ("baz", False),
            ("foo[0].bar.bar", False),
            ("foo[0].baz.bar", False),
            ("foo[0].baz.bar.bix", False),
        ]
        for test, expected in tests:
            regex = rule.regex()
            self.assertEqual(
                re.match(regex, test) is not None,
                expected,
                msg=f"Test: {test} with regex {regex}",
            )

    def test_selector_regex_9(self):
        rule = ClassSelector("foo")[slice(0, 2)] + ClassSelector("bar")
        tests = [
            ("foo[0].bar", True),
            ("foo[0].bar[0]", True),
            ("foo[1].bar", True),
            ("foo[2].bar", False),
            ("foo[0]", False),
            ("bar", False),
            ("baz", False),
            ("foo[0].bar.bar", False),
            ("foo[0].baz.bar", False),
            ("foo[0].baz.bar.bix", False),
        ]
        for test, expected in tests:
            regex = rule.regex()
            self.assertEqual(
                re.match(regex, test) is not None,
                expected,
                msg=f"Test: {test} with regex {regex}",
            )

    def test_selector_regex_10(self):
        rule = ClassSelector("foo")[slice(0, -2, 1), 4] + ClassSelector("bar")
        tests = [
            ("foo[0].bar", True),
            ("foo[0].bar[0]", True),
            ("foo[1].bar", True),
            ("foo[2].bar", False),
        ]
        for test, expected in tests:
            regex = rule.regex()
            self.assertEqual(
                re.match(regex, test) is not None,
                expected,
                msg=f"Test: {test} with regex {regex}",
            )

    def test_selector_regex_11(self):
        rule = WildcardSelector() + ClassSelector("bar")
        tests = [
            ("foo[0].bar", True),
        ]
        for test, expected in tests:
            regex = rule.regex()
            self.assertEqual(
                re.match(regex, test) is not None,
                expected,
                msg=f"Test: {test} with regex {regex}",
            )

    def test_config(self):
        # Example usage:
        config = (
            Config()
            .__.my_attr_1("foo")
            .my_attr_1.attr_1("bar")
            .my_subconfig.attr_1("baz")
        )

        self.assertEqual(config.get("my_attr_1"), "foo")
        self.assertEqual(config.get("my_attr_1.attr_1"), "bar")
        self.assertEqual(config.get("my_subconfig.my_attr_1"), "foo")
        self.assertEqual(config.get("my_subconfig.attr_1"), "baz")

        # Non-existent attributes return None
        self.assertEqual(config.get("my_nonexistent_attr", default=None), None)
        self.assertEqual(config.get("my_nonexistent_attr.attr_1", default=None), None)

        # Non-existent subconfigs inherit from parent
        self.assertEqual(config.get("my_nonexistent_subconfig.my_attr_1"), "foo")

    def test_config_with_selector(self):
        config = (
            Config()
            .__(my_attr_1="foo")
            .my_attr_1(attr_1="bar")
            .my_subconfig(attr_1="baz")
            .my_subconfig(attr_2="bix")
            .my_subconfig.sub_config_2(attr_2="qux")
        )

        config_with_selector = config.with_selector("my_subconfig.sub_config_2")

        parameters = config_with_selector.get_parameters()
        self.assertEqual(parameters.get("attr_1"), None)
        self.assertEqual(parameters.get("attr_2"), "qux")
        self.assertEqual(parameters.get("my_attr_1"), "foo")

    def test_get_parameters(self):
        config = (
            Config()
            .value("foo")
            .block.layer.value("bar")
            .block.layer.block.layer.value("baz")
        )

        parameters = config.get_parameters()
        self.assertEqual(parameters.get("value"), "foo")

        parameters = config.with_selector("block.layer").get_parameters()
        self.assertEqual(parameters.get("value"), "bar")

        parameters = config.with_selector("block.layer.block.layer").get_parameters()
        self.assertEqual(parameters.get("value"), "baz")

    def test_populate(self):
        config = (
            Config()
            .foo(value=1)
            .foo[0:2]
            .populate("value", [2, 3])
            .bar.populate("value", [4, 5])
            .baz.populate("value", lambda i: i, 2)
            .bix[:-2]
            .populate("value", lambda i: i, 4)
            .bix[-2:]
            .populate("value", lambda i: i + 2, 4)
        )

        self.assertEqual(config.foo[0].get("value"), 2)
        self.assertEqual(config.foo[1].get("value"), 3)
        self.assertEqual(config.foo[2].get("value"), 1)
        self.assertEqual(config.bar[0].get("value"), 4)
        self.assertEqual(config.bar[1].get("value"), 5)
        self.assertEqual(config.baz[0].get("value"), 0)
        self.assertEqual(config.baz[1].get("value"), 1)
        self.assertEqual(config.bix[0].get("value"), 0)
        self.assertEqual(config.bix[1].get("value"), 1)
        self.assertEqual(config.bix[2].get("value"), 4)
        self.assertEqual(config.bix[3].get("value"), 5)

    def test_indexed_selector_key(self):
        config = Config().value[0](1)
        # And not value[0]
        self.assertEqual(config._rules[0].key, "value")

    def test_Ref_1(self):
        config = Config().foo(value=1).bar(value=Ref("foo.value"))

        self.assertEqual(config.foo.get("value"), 1)
        self.assertEqual(config.bar.get("value"), 1)

    def test_Ref_2(self):
        config = Config().foo(1).bar(value=Ref("foo"))

        self.assertEqual(config.get("foo"), 1)
        self.assertEqual(config.get("bar.value"), 1)
        self.assertEqual(config.bar.get("value"), 1)

    def test_Ref_3(self):
        config_1 = Config().foo(1).bar(value=Ref("foo"))

        config_2 = Config().merge("baz", config_1)

        self.assertEqual(config_2.get("baz.foo"), 1)
        self.assertEqual(config_2.get("baz.bar.value"), 1)
        self.assertEqual(config_2.baz.get("bar.value"), 1)
        self.assertEqual(config_2.baz.bar.get("value"), 1)

    def test_Ref_4(self):
        config = Config().foo(1).bar(value=Ref("foo", lambda x: x + 1))

        self.assertEqual(config.get("foo"), 1)
        self.assertEqual(config.get("bar.value"), 2)

    def test_merge_default(self):
        config_1 = Config().foo(1).bar(2)

        config_2 = Config().foo(2).bar(3)

        config_prepend_false = (
            Config()
            .merge("baz", config_1, prepend=False)
            .merge("baz", config_2, prepend=False)
        )
        config_prepend_true = (
            Config()
            .merge("baz", config_1, prepend=True)
            .merge("baz", config_2, prepend=True)
        )

        self.assertEqual(config_prepend_false.get("baz.foo"), 2)
        self.assertEqual(config_prepend_false.get("baz.bar"), 3)
        self.assertEqual(config_prepend_true.get("baz.foo"), 1)
        self.assertEqual(config_prepend_true.get("baz.bar"), 2)

    def test_get_last_indexed(self):
        config = Config().foo[0](0).foo[1](1)

        foo0 = config.foo[0].get(None)
        foo1 = config.foo[1].get(None)
        self.assertEqual(foo0, 0)
        self.assertEqual(foo1, 1)

    def test_create_from_function(self):
        def foo():
            return 1

        config = Config().foo(foo)
        self.assertEqual(config.get_parameters().get("foo"), 1)

    def test_create_from_function_with_arg(self):
        def foo(a):
            return a + 1

        config = Config().foo(foo, a=1)
        self.assertEqual(config.get_parameters().get("foo"), 2)

    def test_create_from_function_nested(self):
        def foo(bar):
            return bar

        def bar():
            return 1

        config = Config().foo(foo, bar=bar)
        self.assertEqual(config.get_parameters().get("foo"), 1)

    def test_create_from_function_nested_2(self):
        def foo(bar):
            return bar + 1

        def bar(baz):
            return baz + 1

        config = Config().foo(foo).foo.bar(bar).foo.bar.baz(1)
        self.assertEqual(config.get_parameters().get("foo"), 3)

    def test_create_from_function_with_list(self):
        def foo(bar):
            return bar[0] + bar[1] + 1

        def bar_1():
            return 1

        def bar_2():
            return 2

        config = Config().foo(foo, bar=[bar_1, bar_2])
        self.assertEqual(config.get_parameters().get("foo"), 4)

    def test_naked_rule(self):
        config = Config()(1)
        self.assertEqual(config.get(None), 1)

    # def test_naked_index_rule(self):
    #     config = Config()[0](1)[1](1)
    #     self.assertEqual(config[0].get(None), 1)
    #     self.assertEqual(config.get(None), [1])
