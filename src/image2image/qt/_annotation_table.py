"""Sortable, filterable qtextra tables with stable annotation identities."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

import qtextra.helpers as hp
from qtextra.utils.table_config import TableConfig
from qtextra.widgets.qt_table_view_check import (
    ComboBoxDelegate,
    MultiColumnSingleValueProxyModel,
    QtCheckableItemModel,
    QtCheckableTableView,
)
from qtpy.QtCore import QItemSelectionModel, QModelIndex, QSortFilterProxyModel, Qt, QTimer
from qtpy.QtGui import QColor, QKeyEvent
from qtpy.QtWidgets import QAbstractItemView, QComboBox, QStyleOptionViewItem, QTableView, QWidget


class _ClassDelegate(ComboBoxDelegate):
    def createEditor(self, parent: QWidget, option: QStyleOptionViewItem, index: QModelIndex) -> QComboBox:
        """Open the class choices immediately and commit the user's selection."""
        editor = super().createEditor(parent, option, index)
        editor.activated.connect(lambda: self.commitData.emit(editor))
        timer = QTimer(editor)
        timer.setSingleShot(True)
        timer.timeout.connect(editor.showPopup)
        timer.start(0)
        return editor


class _TableModel(QtCheckableItemModel):
    config: TableConfig
    editable: set[int]
    disabled: set[tuple[Any, int]]
    tooltips: dict[tuple[Any, int], str]
    colors: dict[Any, str]
    on_edit: Callable[[Any, str, Any], bool] | None = None

    def flags(self, index: QModelIndex) -> Qt.ItemFlag:
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags
        flags = Qt.ItemFlag.ItemIsEnabled
        if self.config[index.column()]["selectable"]:
            flags |= Qt.ItemFlag.ItemIsSelectable
        key = self._table[index.row()][-1]
        if (key, index.column()) in self.disabled:
            return flags
        if index.column() in self.editable:
            flags |= (
                Qt.ItemFlag.ItemIsUserCheckable
                if index.column() in self.checkable_columns
                else Qt.ItemFlag.ItemIsEditable
            )
        return flags

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid():
            return None
        value, key = self._table[index.row()][index.column()], self._table[index.row()][-1]
        if role == Qt.ItemDataRole.ToolTipRole:
            return self.tooltips.get((key, index.column()), str(value))
        if (
            role == Qt.ItemDataRole.DecorationRole
            and self.config[index.column()]["tag"] == "class"
            and key in self.colors
        ):
            return QColor(self.colors[key])
        if role == Qt.ItemDataRole.EditRole:
            return value
        return super().data(index, role)

    def headerData(self, column: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.ToolTipRole:
            return self.config[column]["tooltip"]
        return super().headerData(column, orientation, role)

    def setData(self, index: QModelIndex, value: Any, role: int = Qt.ItemDataRole.EditRole) -> bool:
        if not index.isValid():
            return False
        if index.column() not in self.editable or (self._table[index.row()][-1], index.column()) in self.disabled:
            return False
        if role == Qt.ItemDataRole.CheckStateRole:
            value = value is True or getattr(value, "value", value) == getattr(
                Qt.CheckState.Checked, "value", Qt.CheckState.Checked
            )
        if self._table[index.row()][index.column()] == value:
            return False
        return bool(
            self.on_edit and self.on_edit(self._table[index.row()][-1], self.config[index.column()]["tag"], value)
        )


class _TableProxy(MultiColumnSingleValueProxyModel):
    def lessThan(self, left: QModelIndex, right: QModelIndex) -> bool:
        a, b = left.data(self.sortRole()), right.data(self.sortRole())
        if isinstance(a, str) and isinstance(b, str):
            return tuple(
                int(part) if part.isdecimal() else part.casefold() for part in re.split(r"([0-9]+)", a)
            ) < tuple(int(part) if part.isdecimal() else part.casefold() for part in re.split(r"([0-9]+)", b))
        return super().lessThan(left, right)

    def sort(self, column: int, order: Qt.SortOrder = Qt.SortOrder.AscendingOrder) -> None:
        # Sort the proxy so Qt can retain selections on the same source records.
        if column not in self.sourceModel().no_sort_columns:
            QSortFilterProxyModel.sort(self, column, order)

    def filterAcceptsRow(self, row: int, parent: QModelIndex) -> bool:
        model = self.sourceModel()
        for column, query in self.filters_by_text.items():
            columns = range(model.columnCount() - 1) if column == -1 else [column]
            if not any(query.casefold() in str(model.get_data()[row][col]).casefold() for col in columns):
                return False
        return True


class AnnotationTable(QtCheckableTableView):
    """Display bounded rows, reserving the last hidden column for a stable record key."""

    def __init__(
        self,
        parent: QWidget,
        config: TableConfig,
        *,
        paged: bool = False,
        editable: set[str] | None = None,
        on_edit: Callable[[Any, str, Any], bool] | None = None,
    ) -> None:
        columns = config.header
        config.add("Record", "record", hidden=True, show=False, selectable=False, no_sort=True, sizing="fixed")
        self.config = config
        super().__init__(parent, config=config, enable_all_check=False, checkable=False)
        model = _TableModel(
            self,
            [],
            header=config.header,
            hidden_columns=config.hidden_columns,
            checkable_columns=config.checkable_columns,
        )
        model.config = config
        model.editable = {config.find_col_id(tag) for tag in (editable or set())}
        model.disabled, model.tooltips, model.colors = set(), {}, {}
        model.on_edit = on_edit
        self.set_model(model)
        self.proxy = _TableProxy(self)
        self.proxy.setSourceModel(model)
        self.proxy.setSortRole(Qt.ItemDataRole.EditRole)
        model.table_proxy = self.proxy
        self.setModel(self.proxy)
        self.init()
        for index, column in config.column_iter():
            if column["width"]:
                self.setColumnWidth(index, column["width"])
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked | QAbstractItemView.EditTrigger.EditKeyPressed)
        self.header.setSortIndicator(-1, Qt.SortOrder.AscendingOrder)
        self.filter_widget = QWidget(parent)
        self.filter_column = hp.make_combobox(self.filter_widget, ["All columns", *columns])
        self.filter_edit = hp.make_line_edit(
            self.filter_widget, placeholder="Filter current page" if paged else "Filter rows"
        )
        hp.make_h_layout(self.filter_edit, self.filter_column, parent=self.filter_widget, margin=0, stretch_id=0)
        self.filter_edit.textChanged.connect(self._filter)
        self.filter_column.currentIndexChanged.connect(self._filter)

    def _filter(self, *_args: Any) -> None:
        self.proxy.filters_by_text.clear()
        self.proxy.setFilterByColumn(self.filter_edit.text(), self.filter_column.currentIndex() - 1)

    def on_table_clicked(self, index: QModelIndex | None = None) -> None:
        """Keep Qt's native Ctrl/Shift selection instead of replacing it on every click."""

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Preserve multiple selected rows while handling ordinary table navigation."""
        QTableView.keyPressEvent(self, event)

    def sortByColumn(self, column: int, order: Qt.SortOrder | None = None) -> None:
        """Sort the proxy using the requested or current header direction."""
        QTableView.sortByColumn(self, column, order if order is not None else self.header.sortIndicatorOrder())

    def key_at(self, row: int) -> Any:
        """Resolve a displayed row to its stable key after filtering and sorting."""
        source = self.proxy.mapToSource(self.proxy.index(row, 0))
        if not source.isValid():
            raise ValueError("Select a table row first.")
        return self.model().get_data()[source.row()][-1]

    def selected_keys(self) -> list[Any]:
        """Return the keys of all selected visible rows."""
        return [self.key_at(index.row()) for index in self.selectionModel().selectedRows()]

    def set_rows(self, rows: list[list[Any]], *, selected: list[Any] | None = None) -> None:
        """Replace bounded rows while retaining proxy filters, sorting, and selected identities."""
        model = self.model()
        model.beginResetModel()
        model._table = rows
        model.original_index = list(range(len(rows)))
        model._clear_sort_cache()
        model._rebuild_sort_lookup()
        model.endResetModel()
        for row in range(self.proxy.rowCount()):
            if self.key_at(row) in (selected or []):
                self.selectionModel().select(
                    self.proxy.index(row, 0),
                    QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows,
                )

    def set_class_choices(self, labels: list[str]) -> None:
        """Offer valid shared classes when editing the Class column."""
        column = self.config.find_col_id("class")
        delegate = self.itemDelegateForColumn(column)
        if isinstance(delegate, _ClassDelegate):
            delegate.items = list(labels)
        else:
            self.setItemDelegateForColumn(column, _ClassDelegate(labels, self))
