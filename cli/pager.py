"""Pager support for ``hlplot ... --help`` output.

Routes every command/group's ``--help`` text through an interactive pager
(like ``less``) when stdout is a TTY, and passes the text straight through when
piped or captured (e.g. ``hlplot plot --help | cat`` or pytest's ``CliRunner``).
The TTY check is done internally by ``click.echo_via_pager`` -> the pager only
spawns when both stdin and stdout are TTYs.

We override ``get_help_option`` rather than ``get_help`` so that ``get_help()``
stays a pure text accessor (it is also called for the bare ``hlplot`` invocation
and internally by Click); only the ``--help`` flag's behaviour changes.
"""

import typing as t

import click


class _PagerHelpMixin:
    """Mixin that makes the ``--help`` flag page its output via ``less``."""

    def get_help_option(self, ctx: click.Context) -> t.Optional[click.Option]:
        # Start from Click's stock help option so we inherit the flag names,
        # eagerness, expose_value, etc., then swap in a paging callback.
        help_option = super().get_help_option(ctx)  # type: ignore[misc]
        if help_option is None:
            return None

        def show_help(ctx: click.Context, param: click.Parameter, value: bool) -> None:
            if value and not ctx.resilient_parsing:
                click.echo_via_pager(ctx.get_help(), color=ctx.color)
                ctx.exit()

        help_option.callback = show_help
        return help_option


class PagerCommand(_PagerHelpMixin, click.Command):
    """A ``click.Command`` whose ``--help`` pages through ``less`` on a TTY."""


class PagerGroup(_PagerHelpMixin, click.Group):
    """A ``click.Group`` whose ``--help`` pages, and whose subcommands inherit
    the paging behaviour automatically.

    Setting ``command_class`` / ``group_class`` means every subcommand created
    via ``@group.command()`` / ``@group.group()`` (without an explicit ``cls``)
    is built as a ``PagerCommand`` / ``PagerGroup``.
    """

    command_class = PagerCommand
    group_class = type  # -> type(self) == PagerGroup, so nested groups inherit
