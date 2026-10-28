"""Rename a column, or remove columns.

SMAP's ``Process/Modify/RenameFields`` and ``RemoveFields`` in one plugin,
an *operation* choice: a column named as another program wants it, a
column a step wrote under a name one regrets, columns nobody reads that
make every save and every merge of blinks slower.

The work is `smappy.table_edit`, which also says what a column name is tied
to and which of those follow it.  What a rename leaves broken -- a chain
step, a saved setting, a region table naming the old column -- fails in
that plugin with its own error and is the user's to change: a rename has no
way to know which of them meant this column.  Departures from SMAP: a
rename onto an existing column is refused (SMAP moved it to ``<name>_old``)
and there is no *keep old*; the columns the program finds by name are
refused (`table_edit.STRUCTURAL`), where SMAP let positions be renamed
away; and recipes, kept filter bounds and the layers' colour column follow
the rename, where SMAP dropped the layers' filter on it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from . import Context, FormState, Plugin, Result, param, register


def _no_column():
    return [("", "-")]


@dataclass
class EditColumnsSettings:
    operation: str = param("rename", label="operation",
                           choices=[("rename", "rename a column"),
                                    ("remove", "remove columns")],
                           help="rename one column, or remove the ticked ones")
    column: str = param("", label="column", choices=_no_column,
                        help="the column to rename")
    new_name: str = param("", label="new name",
                          help="letters, digits and underscores, as an "
                               "expression can read it")
    remove: List[str] = param(default_factory=list, label="remove",
                              kind="columns", empty_is_all=False,
                              help="the columns to remove; positions, frame, "
                                   "file and the grouping columns are not offered")


def _filters_lost(ctx: Context, names) -> List[str]:
    """The layers' filter bounds on the columns removed: a layer drops a
    bound on a column its table does not have without a word, and what it
    shows grows."""
    if ctx.session is None:
        return []
    out = []
    for i, layer in enumerate(ctx.session.layers):
        if layer.is_image:
            continue
        hit = [n for n in names if n in layer.ranges]
        if hit:
            out.append(f"layer {i + 1} loses its filter on {', '.join(hit)}")
    return out


@register("Analysis/Process/Edit columns")
class EditColumns(Plugin):
    """Rename a column of the localizations, or remove columns."""

    Settings = EditColumnsSettings
    changes_locs = True
    # a column of the localizations; the merged table is made from them
    grouping = "ungrouped"
    version = "1"

    def describe(self, ctx: Optional[Context], settings: EditColumnsSettings) -> FormState:
        """The fields of the chosen operation, and the columns that may be
        edited: the table's own, less those the program finds by name."""
        rename = settings.operation == "rename"
        state = FormState(shown={"column": rename, "new_name": rename,
                                 "remove": not rename})
        if ctx is not None and ctx.session is not None and len(ctx.locs):
            from ..table_edit import editable
            names = editable(ctx.locs)
            state.choices["column"] = [(n, n) for n in names] or _no_column()
            state.choices["remove"] = names
        return state

    def run(self, ctx: Context, settings: EditColumnsSettings) -> Result:
        from ..table_edit import remove_columns, rename_column
        if settings.operation == "rename":
            if not settings.column:
                raise ValueError("choose the column to rename")
            locs, notes = rename_column(ctx.locs, settings.column, settings.new_name)
            new = settings.new_name.strip()
            text = f"{settings.column} renamed {new}"
            renamed = {settings.column: new}
        elif settings.operation == "remove":
            if not settings.remove:
                raise ValueError("tick the columns to remove")
            locs, notes = remove_columns(ctx.locs, settings.remove)
            notes += _filters_lost(ctx, settings.remove)
            for line in notes:
                ctx.warn(line)
            text = f"removed {', '.join(settings.remove)}"
            renamed = {}
        else:
            raise ValueError(f"no operation {settings.operation!r}: rename or remove")
        if notes:
            text += "; " + "; ".join(notes)
        return Result(locs=locs, settings=settings, text=text,
                      renamed_columns=renamed)
