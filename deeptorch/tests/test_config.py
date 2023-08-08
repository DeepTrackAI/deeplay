import unittest
from ..config import Config, selector_matches

class TestConfig(unittest.TestCase):
    
    def test_selector_matches_1(self):
        self.assertTrue(selector_matches(("foo", "bar"), ("foo", "bar")))

    def test_selector_matches_2(self):
        self.assertTrue(selector_matches(("foo", "bar"), ("bar",)))

    def test_selector_matches_3(self):
        self.assertFalse(selector_matches(("foo", "bar"), ("baz",)))

    def test_selector_matches_4(self):
        self.assertFalse(selector_matches(("foo", "bar"), ("foo", "baz")))

    def test_selector_matches_5(self):
        self.assertFalse(selector_matches(("foo", "bar"), ("foo", "bar", "baz")))

    def test_selector_matches_6(self):
        self.assertTrue(selector_matches(("foo", "bar", "baz"), ("foo", "baz")))

    def test_selector_matches_6(self):
        self.assertFalse(selector_matches(("foo", "bar", "baz"), ("baz", "foo")))

    def test_selector_matches_7(self):
        self.assertFalse(selector_matches((), ("foo")))

    def test_selector_matches_8(self):
        self.assertTrue(selector_matches((), ()))
    
    def test_selector_matches_9(self):
        self.assertTrue(selector_matches(("foo",), ()))

    def test_selector_matches_10(self):
        self.assertTrue(selector_matches(("foo", "bar", "foo", "bar"), ("foo", "bar", "foo", "bar")))

    def test_config(self):
        # Example usage:
        config = (
            Config()
                .my_attr_1("foo", True)
                .my_attr_1.attr_1("bar")
                .my_subconfig.attr_1("baz")
        
        )
        
        self.assertEqual(config.get("my_attr_1"), "foo")
        self.assertEqual(config.get("my_attr_1.attr_1"), "bar")
        self.assertEqual(config.get("my_subconfig.my_attr_1"), "foo")
        self.assertEqual(config.get("my_subconfig.attr_1"), "baz")
        
        # Non-existent attributes return None
        self.assertEqual(config.get("my_nonexistent_attr"), None)
        self.assertEqual(config.get("my_nonexistent_attr.attr_1"), None)

        # Non-existent subconfigs inherit from parent
        self.assertEqual(config.get("my_nonexistent_subconfig.my_attr_1"), "foo")

    def test_config_with_selector(self):

        config = (
            Config()
                .my_attr_1("foo", inheritable=True)
                .my_attr_1.attr_1("bar")
                .my_subconfig.attr_1("baz")
                .my_subconfig.attr_2("bix")
                .my_subconfig.sub_config_2.attr_2("qux")
        
        )

        config_with_selector = config.with_selector("my_subconfig")

        self.assertEqual(config_with_selector._selector, ("my_subconfig",))

        config_with_selector = config_with_selector.with_selector("sub_config_2")
        self.assertEqual(config_with_selector._selector, ("my_subconfig", "sub_config_2"))

        
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

        print(config._rules)
        parameters = config.with_selector("block.layer.block.layer").get_parameters()
        self.assertEqual(parameters.get("value"), "baz")