"""Offline school-condition regressions; no game process is accessed."""
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

from lark import Lark
from lark.exceptions import VisitError
from wizwalker.errors import MemoryReadError
from wizwalker.memory.memory_objects.conditionals import school_id_to_names
from wizwalker.extensions.wizsprinter.combat_backends.combat_api import (
    AllCondition, ComparisonOp, TargetType,
)
from wizwalker.extensions.wizsprinter.combat_backends.combat_config_parser import (
    TreeToConfig, get_sprinty_grammar,
)
from wizwalker.extensions.wizsprinter.sprinty_combat import SprintyCombat

PARSER = Lark(get_sprinty_grammar())

def config(text):
    return TreeToConfig().transform(PARSER.parse(text))

def condition(expression):
    return config(f'?({expression}) pass').infinite_rounds[0].priorities[0].condition

def member(school='Fire', health=100):
    school_id = school_id_to_names[school] if isinstance(school, str) else school
    participant = SimpleNamespace(primary_magic_school_id=AsyncMock(return_value=school_id))
    return SimpleNamespace(get_participant=AsyncMock(return_value=participant),
                           health=AsyncMock(return_value=health),
                           max_health=AsyncMock(return_value=1000))

class SchoolParserTests(unittest.TestCase):
    def test_known_names_and_case(self):
        for name, school_id in school_id_to_names.items():
            for spelling in (name.lower(), name.upper(), name):
                with self.subTest(school=spelling):
                    parsed = condition(f'enemy.school == {spelling}')
                    self.assertEqual(parsed.value, school_id)
                    self.assertIs(parsed.op, ComparisonOp.eq)
                    self.assertFalse(parsed.is_percent)

    def test_targets_and_index(self):
        for target in ('boss', 'enemy', 'enemy(1)', 'self', 'ally', 'ally(2)',
                       'any(enemies)', 'all(enemies)', 'any(allies)', 'all(allies)'):
            with self.subTest(target=target):
                self.assertEqual(condition(f'{target}.school != storm').value, school_id_to_names['Storm'])
        indexed = condition('enemy(1).school == fire')
        self.assertIs(indexed.target.target_type, TargetType.type_enemy)
        self.assertEqual(indexed.target.index, 1)

    def test_mixed_health_and_school_priorities(self):
        result = config('?(boss.school == fire && boss.health < 50%) pass | any<damage> @ enemy')
        priorities = result.infinite_rounds[0].priorities
        self.assertEqual(len(priorities), 2)
        self.assertIsInstance(priorities[0].condition, AllCondition)
        health = priorities[0].condition.clauses[1]
        self.assertEqual(health.value, 50.0)
        self.assertTrue(health.is_percent)
        self.assertIsNone(priorities[1].condition)

    def test_invalid_comparisons_are_configuration_errors(self):
        cases = {
            'enemy.school == firre': 'Unknown school',
            'enemy.school == 2343174': 'school name',
            'enemy.school == fire%': 'without %',
            'enemy.school == 50%': 'school name',
            'avg(enemies).school == fire': 'not avg',
            'avg(allies).school != fire': 'not avg',
            'enemy.health == fire': 'numeric',
        }
        cases.update({f'enemy.school {op} fire': 'only supports' for op in ('<', '<=', '>', '>=')})
        for expression, message in cases.items():
            with self.subTest(expression=expression), self.assertRaises(VisitError) as raised:
                condition(expression)
            self.assertIsInstance(raised.exception.orig_exc, ValueError)
            self.assertIn(message, str(raised.exception.orig_exc))

    def test_existing_numeric_conditions(self):
        for expression, value, percent in (
            ('self.health < 25%', 25, True), ('enemy.health >= 2000', 2000, False),
            ('self.power_pips == 3', 3, False), ('avg(enemies).health < 3.5', 3.5, False),
            ('self.beneficial_charms < 1', 1, False),
        ):
            with self.subTest(expression=expression):
                parsed = condition(expression)
                self.assertEqual(parsed.value, value)
                self.assertEqual(parsed.is_percent, percent)

class SchoolEvaluationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.combat = SprintyCombat(SimpleNamespace(), None)
        self.fire, self.ice = member('Fire'), member('Ice')
        self.combat.get_enemies = AsyncMock(return_value=[self.fire, self.ice])
        self.combat.get_boss_or_none = AsyncMock(return_value=self.fire)
        self.combat.get_client_member = AsyncMock(return_value=self.ice)
        self.combat.get_allies = AsyncMock(return_value=[self.fire])

    async def evaluate(self, text):
        return await self.combat.evaluate_condition(condition(text))

    async def test_single_member_uses_participant_school(self):
        self.assertTrue(await self.evaluate('enemy.school == fire'))
        self.assertFalse(await self.evaluate('enemy.school != fire'))
        self.assertTrue(await self.evaluate('enemy.school != storm'))
        self.fire.get_participant.assert_awaited()
        self.fire.get_participant.return_value.primary_magic_school_id.assert_awaited()

    async def test_boss_indexed_enemy_and_allies(self):
        for expression in ('boss.school == fire', 'enemy(1).school == ice',
                           'self.school == ice', 'ally.school == fire',
                           'any(allies).school == fire', 'all(allies).school != storm'):
            with self.subTest(expression=expression):
                self.assertTrue(await self.evaluate(expression))

    async def test_any_and_all_mixed_group(self):
        for expression, expected in (
            ('any(enemies).school == ice', True), ('all(enemies).school == fire', False),
            ('all(enemies).school != storm', True), ('any(enemies).school != fire', True),
            ('all(enemies).school != fire', False),
        ):
            with self.subTest(expression=expression):
                self.assertEqual(await self.evaluate(expression), expected)

    async def test_empty_groups_and_missing_targets(self):
        self.combat.get_enemies.return_value = []
        self.combat.get_boss_or_none.return_value = None
        for target in ('enemy', 'enemy(1)', 'boss', 'any(enemies)', 'all(enemies)'):
            for op in ('==', '!='):
                with self.subTest(target=target, op=op):
                    self.assertFalse(await self.evaluate(f'{target}.school {op} fire'))

    async def test_unknown_school_never_matches(self):
        for school_id in (0, -1, 987654321):
            self.combat.get_enemies.return_value = [member(school_id)]
            for op in ('==', '!='):
                with self.subTest(school_id=school_id, op=op):
                    self.assertFalse(await self.evaluate(f'enemy.school {op} fire'))

    async def test_unknown_member_group_semantics(self):
        self.combat.get_enemies.return_value = [member(0), self.fire]
        self.assertTrue(await self.evaluate('any(enemies).school == fire'))
        self.assertFalse(await self.evaluate('all(enemies).school != storm'))

    async def test_missing_participant_and_memory_read_failure(self):
        self.fire.get_participant.return_value = None
        self.assertFalse(await self.evaluate('enemy.school != storm'))
        self.fire.get_participant.side_effect = MemoryReadError(123)
        self.assertFalse(await self.evaluate('enemy.school != storm'))
        self.fire.get_participant.side_effect = None
        self.fire.get_participant.return_value = SimpleNamespace(
            primary_magic_school_id=AsyncMock(side_effect=MemoryReadError(123)))
        self.assertFalse(await self.evaluate('enemy.school == fire'))

    async def test_school_is_read_again_each_evaluation(self):
        self.assertTrue(await self.evaluate('enemy.school == fire'))
        self.fire.get_participant.return_value.primary_magic_school_id.return_value = school_id_to_names['Ice']
        self.assertFalse(await self.evaluate('enemy.school == fire'))
        self.assertTrue(await self.evaluate('enemy.school == ice'))

    async def test_and_with_health(self):
        self.assertTrue(await self.evaluate('boss.school == fire && boss.health < 25%'))
        self.assertFalse(await self.evaluate('boss.school == ice && boss.health < 25%'))
        self.assertFalse(await self.evaluate('boss.school == fire && boss.health > 25%'))

    async def test_false_school_priority_does_not_attempt_card_selection(self):
        priority = config('?(enemy.school == storm) any<damage> @ enemy').infinite_rounds[0].priorities[0]
        self.combat.try_get_spell = AsyncMock()
        self.assertFalse(await self.combat.try_execute_config(priority))
        self.combat.try_get_spell.assert_not_awaited()

    async def test_matching_school_selects_priority_and_other_school_uses_fallback(self):
        priorities = config('?(enemy.school == fire) pass | pass').infinite_rounds[0].priorities
        self.combat.pass_button = AsyncMock()
        for enemy, expected_priority in ((self.fire, 0), (self.ice, 1)):
            self.combat.get_enemies.return_value = [enemy]
            selected = None
            for index, priority in enumerate(priorities):
                if await self.combat.try_execute_config(priority):
                    selected = index
                    break
            self.assertEqual(selected, expected_priority)
        self.assertEqual(self.combat.pass_button.await_count, 2)

if __name__ == '__main__':
    unittest.main(verbosity=2)
