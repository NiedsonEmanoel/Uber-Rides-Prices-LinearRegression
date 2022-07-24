# Uber Rides Prices LinearRegression
 Regressão linear para a criação de um modelo em que precifica o valor das corridas por aplicativo.

# Notebook:

<!DOCTYPE html>
<html><head><meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">

<title>app</title><script src="./Note_files/require.min.js.download"></script>




<style type="text/css">
    pre { line-height: 125%; }
td.linenos .normal { color: inherit; background-color: transparent; padding-left: 5px; padding-right: 5px; }
span.linenos { color: inherit; background-color: transparent; padding-left: 5px; padding-right: 5px; }
td.linenos .special { color: #000000; background-color: #ffffc0; padding-left: 5px; padding-right: 5px; }
span.linenos.special { color: #000000; background-color: #ffffc0; padding-left: 5px; padding-right: 5px; }
.highlight .hll { background-color: var(--jp-cell-editor-active-background) }
.highlight { background: var(--jp-cell-editor-background); color: var(--jp-mirror-editor-variable-color) }
.highlight .c { color: var(--jp-mirror-editor-comment-color); font-style: italic } /* Comment */
.highlight .err { color: var(--jp-mirror-editor-error-color) } /* Error */
.highlight .k { color: var(--jp-mirror-editor-keyword-color); font-weight: bold } /* Keyword */
.highlight .o { color: var(--jp-mirror-editor-operator-color); font-weight: bold } /* Operator */
.highlight .p { color: var(--jp-mirror-editor-punctuation-color) } /* Punctuation */
.highlight .ch { color: var(--jp-mirror-editor-comment-color); font-style: italic } /* Comment.Hashbang */
.highlight .cm { color: var(--jp-mirror-editor-comment-color); font-style: italic } /* Comment.Multiline */
.highlight .cp { color: var(--jp-mirror-editor-comment-color); font-style: italic } /* Comment.Preproc */
.highlight .cpf { color: var(--jp-mirror-editor-comment-color); font-style: italic } /* Comment.PreprocFile */
.highlight .c1 { color: var(--jp-mirror-editor-comment-color); font-style: italic } /* Comment.Single */
.highlight .cs { color: var(--jp-mirror-editor-comment-color); font-style: italic } /* Comment.Special */
.highlight .kc { color: var(--jp-mirror-editor-keyword-color); font-weight: bold } /* Keyword.Constant */
.highlight .kd { color: var(--jp-mirror-editor-keyword-color); font-weight: bold } /* Keyword.Declaration */
.highlight .kn { color: var(--jp-mirror-editor-keyword-color); font-weight: bold } /* Keyword.Namespace */
.highlight .kp { color: var(--jp-mirror-editor-keyword-color); font-weight: bold } /* Keyword.Pseudo */
.highlight .kr { color: var(--jp-mirror-editor-keyword-color); font-weight: bold } /* Keyword.Reserved */
.highlight .kt { color: var(--jp-mirror-editor-keyword-color); font-weight: bold } /* Keyword.Type */
.highlight .m { color: var(--jp-mirror-editor-number-color) } /* Literal.Number */
.highlight .s { color: var(--jp-mirror-editor-string-color) } /* Literal.String */
.highlight .ow { color: var(--jp-mirror-editor-operator-color); font-weight: bold } /* Operator.Word */
.highlight .w { color: var(--jp-mirror-editor-variable-color) } /* Text.Whitespace */
.highlight .mb { color: var(--jp-mirror-editor-number-color) } /* Literal.Number.Bin */
.highlight .mf { color: var(--jp-mirror-editor-number-color) } /* Literal.Number.Float */
.highlight .mh { color: var(--jp-mirror-editor-number-color) } /* Literal.Number.Hex */
.highlight .mi { color: var(--jp-mirror-editor-number-color) } /* Literal.Number.Integer */
.highlight .mo { color: var(--jp-mirror-editor-number-color) } /* Literal.Number.Oct */
.highlight .sa { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Affix */
.highlight .sb { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Backtick */
.highlight .sc { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Char */
.highlight .dl { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Delimiter */
.highlight .sd { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Doc */
.highlight .s2 { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Double */
.highlight .se { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Escape */
.highlight .sh { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Heredoc */
.highlight .si { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Interpol */
.highlight .sx { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Other */
.highlight .sr { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Regex */
.highlight .s1 { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Single */
.highlight .ss { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Symbol */
.highlight .il { color: var(--jp-mirror-editor-number-color) } /* Literal.Number.Integer.Long */
  </style>



<style type="text/css">
/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*
 * Mozilla scrollbar styling
 */

/* use standard opaque scrollbars for most nodes */
[data-jp-theme-scrollbars='true'] {
  scrollbar-color: rgb(var(--jp-scrollbar-thumb-color))
    var(--jp-scrollbar-background-color);
}

/* for code nodes, use a transparent style of scrollbar. These selectors
 * will match lower in the tree, and so will override the above */
[data-jp-theme-scrollbars='true'] .CodeMirror-hscrollbar,
[data-jp-theme-scrollbars='true'] .CodeMirror-vscrollbar {
  scrollbar-color: rgba(var(--jp-scrollbar-thumb-color), 0.5) transparent;
}

/* tiny scrollbar */

.jp-scrollbar-tiny {
  scrollbar-color: rgba(var(--jp-scrollbar-thumb-color), 0.5) transparent;
  scrollbar-width: thin;
}

/*
 * Webkit scrollbar styling
 */

/* use standard opaque scrollbars for most nodes */

[data-jp-theme-scrollbars='true'] ::-webkit-scrollbar,
[data-jp-theme-scrollbars='true'] ::-webkit-scrollbar-corner {
  background: var(--jp-scrollbar-background-color);
}

[data-jp-theme-scrollbars='true'] ::-webkit-scrollbar-thumb {
  background: rgb(var(--jp-scrollbar-thumb-color));
  border: var(--jp-scrollbar-thumb-margin) solid transparent;
  background-clip: content-box;
  border-radius: var(--jp-scrollbar-thumb-radius);
}

[data-jp-theme-scrollbars='true'] ::-webkit-scrollbar-track:horizontal {
  border-left: var(--jp-scrollbar-endpad) solid
    var(--jp-scrollbar-background-color);
  border-right: var(--jp-scrollbar-endpad) solid
    var(--jp-scrollbar-background-color);
}

[data-jp-theme-scrollbars='true'] ::-webkit-scrollbar-track:vertical {
  border-top: var(--jp-scrollbar-endpad) solid
    var(--jp-scrollbar-background-color);
  border-bottom: var(--jp-scrollbar-endpad) solid
    var(--jp-scrollbar-background-color);
}

/* for code nodes, use a transparent style of scrollbar */

[data-jp-theme-scrollbars='true'] .CodeMirror-hscrollbar::-webkit-scrollbar,
[data-jp-theme-scrollbars='true'] .CodeMirror-vscrollbar::-webkit-scrollbar,
[data-jp-theme-scrollbars='true']
  .CodeMirror-hscrollbar::-webkit-scrollbar-corner,
[data-jp-theme-scrollbars='true']
  .CodeMirror-vscrollbar::-webkit-scrollbar-corner {
  background-color: transparent;
}

[data-jp-theme-scrollbars='true']
  .CodeMirror-hscrollbar::-webkit-scrollbar-thumb,
[data-jp-theme-scrollbars='true']
  .CodeMirror-vscrollbar::-webkit-scrollbar-thumb {
  background: rgba(var(--jp-scrollbar-thumb-color), 0.5);
  border: var(--jp-scrollbar-thumb-margin) solid transparent;
  background-clip: content-box;
  border-radius: var(--jp-scrollbar-thumb-radius);
}

[data-jp-theme-scrollbars='true']
  .CodeMirror-hscrollbar::-webkit-scrollbar-track:horizontal {
  border-left: var(--jp-scrollbar-endpad) solid transparent;
  border-right: var(--jp-scrollbar-endpad) solid transparent;
}

[data-jp-theme-scrollbars='true']
  .CodeMirror-vscrollbar::-webkit-scrollbar-track:vertical {
  border-top: var(--jp-scrollbar-endpad) solid transparent;
  border-bottom: var(--jp-scrollbar-endpad) solid transparent;
}

/* tiny scrollbar */

.jp-scrollbar-tiny::-webkit-scrollbar,
.jp-scrollbar-tiny::-webkit-scrollbar-corner {
  background-color: transparent;
  height: 4px;
  width: 4px;
}

.jp-scrollbar-tiny::-webkit-scrollbar-thumb {
  background: rgba(var(--jp-scrollbar-thumb-color), 0.5);
}

.jp-scrollbar-tiny::-webkit-scrollbar-track:horizontal {
  border-left: 0px solid transparent;
  border-right: 0px solid transparent;
}

.jp-scrollbar-tiny::-webkit-scrollbar-track:vertical {
  border-top: 0px solid transparent;
  border-bottom: 0px solid transparent;
}

/*
 * Phosphor
 */

.lm-ScrollBar[data-orientation='horizontal'] {
  min-height: 16px;
  max-height: 16px;
  min-width: 45px;
  border-top: 1px solid #a0a0a0;
}

.lm-ScrollBar[data-orientation='vertical'] {
  min-width: 16px;
  max-width: 16px;
  min-height: 45px;
  border-left: 1px solid #a0a0a0;
}

.lm-ScrollBar-button {
  background-color: #f0f0f0;
  background-position: center center;
  min-height: 15px;
  max-height: 15px;
  min-width: 15px;
  max-width: 15px;
}

.lm-ScrollBar-button:hover {
  background-color: #dadada;
}

.lm-ScrollBar-button.lm-mod-active {
  background-color: #cdcdcd;
}

.lm-ScrollBar-track {
  background: #f0f0f0;
}

.lm-ScrollBar-thumb {
  background: #cdcdcd;
}

.lm-ScrollBar-thumb:hover {
  background: #bababa;
}

.lm-ScrollBar-thumb.lm-mod-active {
  background: #a0a0a0;
}

.lm-ScrollBar[data-orientation='horizontal'] .lm-ScrollBar-thumb {
  height: 100%;
  min-width: 15px;
  border-left: 1px solid #a0a0a0;
  border-right: 1px solid #a0a0a0;
}

.lm-ScrollBar[data-orientation='vertical'] .lm-ScrollBar-thumb {
  width: 100%;
  min-height: 15px;
  border-top: 1px solid #a0a0a0;
  border-bottom: 1px solid #a0a0a0;
}

.lm-ScrollBar[data-orientation='horizontal']
  .lm-ScrollBar-button[data-action='decrement'] {
  background-image: var(--jp-icon-caret-left);
  background-size: 17px;
}

.lm-ScrollBar[data-orientation='horizontal']
  .lm-ScrollBar-button[data-action='increment'] {
  background-image: var(--jp-icon-caret-right);
  background-size: 17px;
}

.lm-ScrollBar[data-orientation='vertical']
  .lm-ScrollBar-button[data-action='decrement'] {
  background-image: var(--jp-icon-caret-up);
  background-size: 17px;
}

.lm-ScrollBar[data-orientation='vertical']
  .lm-ScrollBar-button[data-action='increment'] {
  background-image: var(--jp-icon-caret-down);
  background-size: 17px;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/


/* <DEPRECATED> */ .p-Widget, /* </DEPRECATED> */
.lm-Widget {
  box-sizing: border-box;
  position: relative;
  overflow: hidden;
  cursor: default;
}


/* <DEPRECATED> */ .p-Widget.p-mod-hidden, /* </DEPRECATED> */
.lm-Widget.lm-mod-hidden {
  display: none !important;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/


/* <DEPRECATED> */ .p-CommandPalette, /* </DEPRECATED> */
.lm-CommandPalette {
  display: flex;
  flex-direction: column;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}


/* <DEPRECATED> */ .p-CommandPalette-search, /* </DEPRECATED> */
.lm-CommandPalette-search {
  flex: 0 0 auto;
}


/* <DEPRECATED> */ .p-CommandPalette-content, /* </DEPRECATED> */
.lm-CommandPalette-content {
  flex: 1 1 auto;
  margin: 0;
  padding: 0;
  min-height: 0;
  overflow: auto;
  list-style-type: none;
}


/* <DEPRECATED> */ .p-CommandPalette-header, /* </DEPRECATED> */
.lm-CommandPalette-header {
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
}


/* <DEPRECATED> */ .p-CommandPalette-item, /* </DEPRECATED> */
.lm-CommandPalette-item {
  display: flex;
  flex-direction: row;
}


/* <DEPRECATED> */ .p-CommandPalette-itemIcon, /* </DEPRECATED> */
.lm-CommandPalette-itemIcon {
  flex: 0 0 auto;
}


/* <DEPRECATED> */ .p-CommandPalette-itemContent, /* </DEPRECATED> */
.lm-CommandPalette-itemContent {
  flex: 1 1 auto;
  overflow: hidden;
}


/* <DEPRECATED> */ .p-CommandPalette-itemShortcut, /* </DEPRECATED> */
.lm-CommandPalette-itemShortcut {
  flex: 0 0 auto;
}


/* <DEPRECATED> */ .p-CommandPalette-itemLabel, /* </DEPRECATED> */
.lm-CommandPalette-itemLabel {
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
}

.lm-close-icon {
	border:1px solid transparent;
  background-color: transparent;
  position: absolute;
	z-index:1;
	right:3%;
	top: 0;
	bottom: 0;
	margin: auto;
	padding: 7px 0;
	display: none;
	vertical-align: middle;
  outline: 0;
  cursor: pointer;
}
.lm-close-icon:after {
	content: "X";
	display: block;
	width: 15px;
	height: 15px;
	text-align: center;
	color:#000;
	font-weight: normal;
	font-size: 12px;
	cursor: pointer;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/


/* <DEPRECATED> */ .p-DockPanel, /* </DEPRECATED> */
.lm-DockPanel {
  z-index: 0;
}


/* <DEPRECATED> */ .p-DockPanel-widget, /* </DEPRECATED> */
.lm-DockPanel-widget {
  z-index: 0;
}


/* <DEPRECATED> */ .p-DockPanel-tabBar, /* </DEPRECATED> */
.lm-DockPanel-tabBar {
  z-index: 1;
}


/* <DEPRECATED> */ .p-DockPanel-handle, /* </DEPRECATED> */
.lm-DockPanel-handle {
  z-index: 2;
}


/* <DEPRECATED> */ .p-DockPanel-handle.p-mod-hidden, /* </DEPRECATED> */
.lm-DockPanel-handle.lm-mod-hidden {
  display: none !important;
}


/* <DEPRECATED> */ .p-DockPanel-handle:after, /* </DEPRECATED> */
.lm-DockPanel-handle:after {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  content: '';
}


/* <DEPRECATED> */
.p-DockPanel-handle[data-orientation='horizontal'],
/* </DEPRECATED> */
.lm-DockPanel-handle[data-orientation='horizontal'] {
  cursor: ew-resize;
}


/* <DEPRECATED> */
.p-DockPanel-handle[data-orientation='vertical'],
/* </DEPRECATED> */
.lm-DockPanel-handle[data-orientation='vertical'] {
  cursor: ns-resize;
}


/* <DEPRECATED> */
.p-DockPanel-handle[data-orientation='horizontal']:after,
/* </DEPRECATED> */
.lm-DockPanel-handle[data-orientation='horizontal']:after {
  left: 50%;
  min-width: 8px;
  transform: translateX(-50%);
}


/* <DEPRECATED> */
.p-DockPanel-handle[data-orientation='vertical']:after,
/* </DEPRECATED> */
.lm-DockPanel-handle[data-orientation='vertical']:after {
  top: 50%;
  min-height: 8px;
  transform: translateY(-50%);
}


/* <DEPRECATED> */ .p-DockPanel-overlay, /* </DEPRECATED> */
.lm-DockPanel-overlay {
  z-index: 3;
  box-sizing: border-box;
  pointer-events: none;
}


/* <DEPRECATED> */ .p-DockPanel-overlay.p-mod-hidden, /* </DEPRECATED> */
.lm-DockPanel-overlay.lm-mod-hidden {
  display: none !important;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/


/* <DEPRECATED> */ .p-Menu, /* </DEPRECATED> */
.lm-Menu {
  z-index: 10000;
  position: absolute;
  white-space: nowrap;
  overflow-x: hidden;
  overflow-y: auto;
  outline: none;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}


/* <DEPRECATED> */ .p-Menu-content, /* </DEPRECATED> */
.lm-Menu-content {
  margin: 0;
  padding: 0;
  display: table;
  list-style-type: none;
}


/* <DEPRECATED> */ .p-Menu-item, /* </DEPRECATED> */
.lm-Menu-item {
  display: table-row;
}


/* <DEPRECATED> */
.p-Menu-item.p-mod-hidden,
.p-Menu-item.p-mod-collapsed,
/* </DEPRECATED> */
.lm-Menu-item.lm-mod-hidden,
.lm-Menu-item.lm-mod-collapsed {
  display: none !important;
}


/* <DEPRECATED> */
.p-Menu-itemIcon,
.p-Menu-itemSubmenuIcon,
/* </DEPRECATED> */
.lm-Menu-itemIcon,
.lm-Menu-itemSubmenuIcon {
  display: table-cell;
  text-align: center;
}


/* <DEPRECATED> */ .p-Menu-itemLabel, /* </DEPRECATED> */
.lm-Menu-itemLabel {
  display: table-cell;
  text-align: left;
}


/* <DEPRECATED> */ .p-Menu-itemShortcut, /* </DEPRECATED> */
.lm-Menu-itemShortcut {
  display: table-cell;
  text-align: right;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/


/* <DEPRECATED> */ .p-MenuBar, /* </DEPRECATED> */
.lm-MenuBar {
  outline: none;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}


/* <DEPRECATED> */ .p-MenuBar-content, /* </DEPRECATED> */
.lm-MenuBar-content {
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: row;
  list-style-type: none;
}


/* <DEPRECATED> */ .p--MenuBar-item, /* </DEPRECATED> */
.lm-MenuBar-item {
  box-sizing: border-box;
}


/* <DEPRECATED> */
.p-MenuBar-itemIcon,
.p-MenuBar-itemLabel,
/* </DEPRECATED> */
.lm-MenuBar-itemIcon,
.lm-MenuBar-itemLabel {
  display: inline-block;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/


/* <DEPRECATED> */ .p-ScrollBar, /* </DEPRECATED> */
.lm-ScrollBar {
  display: flex;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}


/* <DEPRECATED> */
.p-ScrollBar[data-orientation='horizontal'],
/* </DEPRECATED> */
.lm-ScrollBar[data-orientation='horizontal'] {
  flex-direction: row;
}


/* <DEPRECATED> */
.p-ScrollBar[data-orientation='vertical'],
/* </DEPRECATED> */
.lm-ScrollBar[data-orientation='vertical'] {
  flex-direction: column;
}


/* <DEPRECATED> */ .p-ScrollBar-button, /* </DEPRECATED> */
.lm-ScrollBar-button {
  box-sizing: border-box;
  flex: 0 0 auto;
}


/* <DEPRECATED> */ .p-ScrollBar-track, /* </DEPRECATED> */
.lm-ScrollBar-track {
  box-sizing: border-box;
  position: relative;
  overflow: hidden;
  flex: 1 1 auto;
}


/* <DEPRECATED> */ .p-ScrollBar-thumb, /* </DEPRECATED> */
.lm-ScrollBar-thumb {
  box-sizing: border-box;
  position: absolute;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/


/* <DEPRECATED> */ .p-SplitPanel-child, /* </DEPRECATED> */
.lm-SplitPanel-child {
  z-index: 0;
}


/* <DEPRECATED> */ .p-SplitPanel-handle, /* </DEPRECATED> */
.lm-SplitPanel-handle {
  z-index: 1;
}


/* <DEPRECATED> */ .p-SplitPanel-handle.p-mod-hidden, /* </DEPRECATED> */
.lm-SplitPanel-handle.lm-mod-hidden {
  display: none !important;
}


/* <DEPRECATED> */ .p-SplitPanel-handle:after, /* </DEPRECATED> */
.lm-SplitPanel-handle:after {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  content: '';
}


/* <DEPRECATED> */
.p-SplitPanel[data-orientation='horizontal'] > .p-SplitPanel-handle,
/* </DEPRECATED> */
.lm-SplitPanel[data-orientation='horizontal'] > .lm-SplitPanel-handle {
  cursor: ew-resize;
}


/* <DEPRECATED> */
.p-SplitPanel[data-orientation='vertical'] > .p-SplitPanel-handle,
/* </DEPRECATED> */
.lm-SplitPanel[data-orientation='vertical'] > .lm-SplitPanel-handle {
  cursor: ns-resize;
}


/* <DEPRECATED> */
.p-SplitPanel[data-orientation='horizontal'] > .p-SplitPanel-handle:after,
/* </DEPRECATED> */
.lm-SplitPanel[data-orientation='horizontal'] > .lm-SplitPanel-handle:after {
  left: 50%;
  min-width: 8px;
  transform: translateX(-50%);
}


/* <DEPRECATED> */
.p-SplitPanel[data-orientation='vertical'] > .p-SplitPanel-handle:after,
/* </DEPRECATED> */
.lm-SplitPanel[data-orientation='vertical'] > .lm-SplitPanel-handle:after {
  top: 50%;
  min-height: 8px;
  transform: translateY(-50%);
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/


/* <DEPRECATED> */ .p-TabBar, /* </DEPRECATED> */
.lm-TabBar {
  display: flex;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}


/* <DEPRECATED> */ .p-TabBar[data-orientation='horizontal'], /* </DEPRECATED> */
.lm-TabBar[data-orientation='horizontal'] {
  flex-direction: row;
  align-items: flex-end;
}


/* <DEPRECATED> */ .p-TabBar[data-orientation='vertical'], /* </DEPRECATED> */
.lm-TabBar[data-orientation='vertical'] {
  flex-direction: column;
  align-items: flex-end;
}


/* <DEPRECATED> */ .p-TabBar-content, /* </DEPRECATED> */
.lm-TabBar-content {
  margin: 0;
  padding: 0;
  display: flex;
  flex: 1 1 auto;
  list-style-type: none;
}


/* <DEPRECATED> */
.p-TabBar[data-orientation='horizontal'] > .p-TabBar-content,
/* </DEPRECATED> */
.lm-TabBar[data-orientation='horizontal'] > .lm-TabBar-content {
  flex-direction: row;
}


/* <DEPRECATED> */
.p-TabBar[data-orientation='vertical'] > .p-TabBar-content,
/* </DEPRECATED> */
.lm-TabBar[data-orientation='vertical'] > .lm-TabBar-content {
  flex-direction: column;
}


/* <DEPRECATED> */ .p-TabBar-tab, /* </DEPRECATED> */
.lm-TabBar-tab {
  display: flex;
  flex-direction: row;
  box-sizing: border-box;
  overflow: hidden;
}


/* <DEPRECATED> */
.p-TabBar-tabIcon,
.p-TabBar-tabCloseIcon,
/* </DEPRECATED> */
.lm-TabBar-tabIcon,
.lm-TabBar-tabCloseIcon {
  flex: 0 0 auto;
}


/* <DEPRECATED> */ .p-TabBar-tabLabel, /* </DEPRECATED> */
.lm-TabBar-tabLabel {
  flex: 1 1 auto;
  overflow: hidden;
  white-space: nowrap;
}


.lm-TabBar-tabInput {
  user-select: all;
  width: 100%;
  box-sizing : border-box;
}


/* <DEPRECATED> */ .p-TabBar-tab.p-mod-hidden, /* </DEPRECATED> */
.lm-TabBar-tab.lm-mod-hidden {
  display: none !important;
}


.lm-TabBar-addButton.lm-mod-hidden {
  display: none !important;
}


/* <DEPRECATED> */ .p-TabBar.p-mod-dragging .p-TabBar-tab, /* </DEPRECATED> */
.lm-TabBar.lm-mod-dragging .lm-TabBar-tab {
  position: relative;
}


/* <DEPRECATED> */
.p-TabBar.p-mod-dragging[data-orientation='horizontal'] .p-TabBar-tab,
/* </DEPRECATED> */
.lm-TabBar.lm-mod-dragging[data-orientation='horizontal'] .lm-TabBar-tab {
  left: 0;
  transition: left 150ms ease;
}


/* <DEPRECATED> */
.p-TabBar.p-mod-dragging[data-orientation='vertical'] .p-TabBar-tab,
/* </DEPRECATED> */
.lm-TabBar.lm-mod-dragging[data-orientation='vertical'] .lm-TabBar-tab {
  top: 0;
  transition: top 150ms ease;
}


/* <DEPRECATED> */
.p-TabBar.p-mod-dragging .p-TabBar-tab.p-mod-dragging,
/* </DEPRECATED> */
.lm-TabBar.lm-mod-dragging .lm-TabBar-tab.lm-mod-dragging {
  transition: none;
}

.lm-TabBar-tabLabel .lm-TabBar-tabInput {
  user-select: all;
  width: 100%;
  box-sizing : border-box;
  background: inherit;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/


/* <DEPRECATED> */ .p-TabPanel-tabBar, /* </DEPRECATED> */
.lm-TabPanel-tabBar {
  z-index: 1;
}


/* <DEPRECATED> */ .p-TabPanel-stackedPanel, /* </DEPRECATED> */
.lm-TabPanel-stackedPanel {
  z-index: 0;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/

@charset "UTF-8";
html{
  -webkit-box-sizing:border-box;
          box-sizing:border-box; }

*,
*::before,
*::after{
  -webkit-box-sizing:inherit;
          box-sizing:inherit; }

body{
  font-size:14px;
  font-weight:400;
  letter-spacing:0;
  line-height:1.28581;
  text-transform:none;
  color:#182026;
  font-family:-apple-system, "BlinkMacSystemFont", "Segoe UI", "Roboto", "Oxygen", "Ubuntu", "Cantarell", "Open Sans", "Helvetica Neue", "Icons16", sans-serif; }

p{
  margin-bottom:10px;
  margin-top:0; }

small{
  font-size:12px; }

strong{
  font-weight:600; }

::-moz-selection{
  background:rgba(125, 188, 255, 0.6); }

::selection{
  background:rgba(125, 188, 255, 0.6); }
.bp3-heading{
  color:#182026;
  font-weight:600;
  margin:0 0 10px;
  padding:0; }
  .bp3-dark .bp3-heading{
    color:#f5f8fa; }

h1.bp3-heading, .bp3-running-text h1{
  font-size:36px;
  line-height:40px; }

h2.bp3-heading, .bp3-running-text h2{
  font-size:28px;
  line-height:32px; }

h3.bp3-heading, .bp3-running-text h3{
  font-size:22px;
  line-height:25px; }

h4.bp3-heading, .bp3-running-text h4{
  font-size:18px;
  line-height:21px; }

h5.bp3-heading, .bp3-running-text h5{
  font-size:16px;
  line-height:19px; }

h6.bp3-heading, .bp3-running-text h6{
  font-size:14px;
  line-height:16px; }
.bp3-ui-text{
  font-size:14px;
  font-weight:400;
  letter-spacing:0;
  line-height:1.28581;
  text-transform:none; }

.bp3-monospace-text{
  font-family:monospace;
  text-transform:none; }

.bp3-text-muted{
  color:#5c7080; }
  .bp3-dark .bp3-text-muted{
    color:#a7b6c2; }

.bp3-text-disabled{
  color:rgba(92, 112, 128, 0.6); }
  .bp3-dark .bp3-text-disabled{
    color:rgba(167, 182, 194, 0.6); }

.bp3-text-overflow-ellipsis{
  overflow:hidden;
  text-overflow:ellipsis;
  white-space:nowrap;
  word-wrap:normal; }
.bp3-running-text{
  font-size:14px;
  line-height:1.5; }
  .bp3-running-text h1{
    color:#182026;
    font-weight:600;
    margin-bottom:20px;
    margin-top:40px; }
    .bp3-dark .bp3-running-text h1{
      color:#f5f8fa; }
  .bp3-running-text h2{
    color:#182026;
    font-weight:600;
    margin-bottom:20px;
    margin-top:40px; }
    .bp3-dark .bp3-running-text h2{
      color:#f5f8fa; }
  .bp3-running-text h3{
    color:#182026;
    font-weight:600;
    margin-bottom:20px;
    margin-top:40px; }
    .bp3-dark .bp3-running-text h3{
      color:#f5f8fa; }
  .bp3-running-text h4{
    color:#182026;
    font-weight:600;
    margin-bottom:20px;
    margin-top:40px; }
    .bp3-dark .bp3-running-text h4{
      color:#f5f8fa; }
  .bp3-running-text h5{
    color:#182026;
    font-weight:600;
    margin-bottom:20px;
    margin-top:40px; }
    .bp3-dark .bp3-running-text h5{
      color:#f5f8fa; }
  .bp3-running-text h6{
    color:#182026;
    font-weight:600;
    margin-bottom:20px;
    margin-top:40px; }
    .bp3-dark .bp3-running-text h6{
      color:#f5f8fa; }
  .bp3-running-text hr{
    border:none;
    border-bottom:1px solid rgba(16, 22, 26, 0.15);
    margin:20px 0; }
    .bp3-dark .bp3-running-text hr{
      border-color:rgba(255, 255, 255, 0.15); }
  .bp3-running-text p{
    margin:0 0 10px;
    padding:0; }

.bp3-text-large{
  font-size:16px; }

.bp3-text-small{
  font-size:12px; }
a{
  color:#106ba3;
  text-decoration:none; }
  a:hover{
    color:#106ba3;
    cursor:pointer;
    text-decoration:underline; }
  a .bp3-icon, a .bp3-icon-standard, a .bp3-icon-large{
    color:inherit; }
  a code,
  .bp3-dark a code{
    color:inherit; }
  .bp3-dark a,
  .bp3-dark a:hover{
    color:#48aff0; }
    .bp3-dark a .bp3-icon, .bp3-dark a .bp3-icon-standard, .bp3-dark a .bp3-icon-large,
    .bp3-dark a:hover .bp3-icon,
    .bp3-dark a:hover .bp3-icon-standard,
    .bp3-dark a:hover .bp3-icon-large{
      color:inherit; }
.bp3-running-text code, .bp3-code{
  font-family:monospace;
  text-transform:none;
  background:rgba(255, 255, 255, 0.7);
  border-radius:3px;
  -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2);
          box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2);
  color:#5c7080;
  font-size:smaller;
  padding:2px 5px; }
  .bp3-dark .bp3-running-text code, .bp3-running-text .bp3-dark code, .bp3-dark .bp3-code{
    background:rgba(16, 22, 26, 0.3);
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4);
    color:#a7b6c2; }
  .bp3-running-text a > code, a > .bp3-code{
    color:#137cbd; }
    .bp3-dark .bp3-running-text a > code, .bp3-running-text .bp3-dark a > code, .bp3-dark a > .bp3-code{
      color:inherit; }

.bp3-running-text pre, .bp3-code-block{
  font-family:monospace;
  text-transform:none;
  background:rgba(255, 255, 255, 0.7);
  border-radius:3px;
  -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.15);
          box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.15);
  color:#182026;
  display:block;
  font-size:13px;
  line-height:1.4;
  margin:10px 0;
  padding:13px 15px 12px;
  word-break:break-all;
  word-wrap:break-word; }
  .bp3-dark .bp3-running-text pre, .bp3-running-text .bp3-dark pre, .bp3-dark .bp3-code-block{
    background:rgba(16, 22, 26, 0.3);
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4);
    color:#f5f8fa; }
  .bp3-running-text pre > code, .bp3-code-block > code{
    background:none;
    -webkit-box-shadow:none;
            box-shadow:none;
    color:inherit;
    font-size:inherit;
    padding:0; }

.bp3-running-text kbd, .bp3-key{
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  background:#ffffff;
  border-radius:3px;
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.2);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.2);
  color:#5c7080;
  display:-webkit-inline-box;
  display:-ms-inline-flexbox;
  display:inline-flex;
  font-family:inherit;
  font-size:12px;
  height:24px;
  -webkit-box-pack:center;
      -ms-flex-pack:center;
          justify-content:center;
  line-height:24px;
  min-width:24px;
  padding:3px 6px;
  vertical-align:middle; }
  .bp3-running-text kbd .bp3-icon, .bp3-key .bp3-icon, .bp3-running-text kbd .bp3-icon-standard, .bp3-key .bp3-icon-standard, .bp3-running-text kbd .bp3-icon-large, .bp3-key .bp3-icon-large{
    margin-right:5px; }
  .bp3-dark .bp3-running-text kbd, .bp3-running-text .bp3-dark kbd, .bp3-dark .bp3-key{
    background:#394b59;
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.4);
    color:#a7b6c2; }
.bp3-running-text blockquote, .bp3-blockquote{
  border-left:solid 4px rgba(167, 182, 194, 0.5);
  margin:0 0 10px;
  padding:0 20px; }
  .bp3-dark .bp3-running-text blockquote, .bp3-running-text .bp3-dark blockquote, .bp3-dark .bp3-blockquote{
    border-color:rgba(115, 134, 148, 0.5); }
.bp3-running-text ul,
.bp3-running-text ol, .bp3-list{
  margin:10px 0;
  padding-left:30px; }
  .bp3-running-text ul li:not(:last-child), .bp3-running-text ol li:not(:last-child), .bp3-list li:not(:last-child){
    margin-bottom:5px; }
  .bp3-running-text ul ol, .bp3-running-text ol ol, .bp3-list ol,
  .bp3-running-text ul ul,
  .bp3-running-text ol ul,
  .bp3-list ul{
    margin-top:5px; }

.bp3-list-unstyled{
  list-style:none;
  margin:0;
  padding:0; }
  .bp3-list-unstyled li{
    padding:0; }
.bp3-rtl{
  text-align:right; }

.bp3-dark{
  color:#f5f8fa; }

:focus{
  outline:rgba(19, 124, 189, 0.6) auto 2px;
  outline-offset:2px;
  -moz-outline-radius:6px; }

.bp3-focus-disabled :focus{
  outline:none !important; }
  .bp3-focus-disabled :focus ~ .bp3-control-indicator{
    outline:none !important; }

.bp3-alert{
  max-width:400px;
  padding:20px; }

.bp3-alert-body{
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex; }
  .bp3-alert-body .bp3-icon{
    font-size:40px;
    margin-right:20px;
    margin-top:0; }

.bp3-alert-contents{
  word-break:break-word; }

.bp3-alert-footer{
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-orient:horizontal;
  -webkit-box-direction:reverse;
      -ms-flex-direction:row-reverse;
          flex-direction:row-reverse;
  margin-top:10px; }
  .bp3-alert-footer .bp3-button{
    margin-left:10px; }
.bp3-breadcrumbs{
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  cursor:default;
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -ms-flex-wrap:wrap;
      flex-wrap:wrap;
  height:30px;
  list-style:none;
  margin:0;
  padding:0; }
  .bp3-breadcrumbs > li{
    -webkit-box-align:center;
        -ms-flex-align:center;
            align-items:center;
    display:-webkit-box;
    display:-ms-flexbox;
    display:flex; }
    .bp3-breadcrumbs > li::after{
      background:url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'%3e%3cpath fill-rule='evenodd' clip-rule='evenodd' d='M10.71 7.29l-4-4a1.003 1.003 0 00-1.42 1.42L8.59 8 5.3 11.29c-.19.18-.3.43-.3.71a1.003 1.003 0 001.71.71l4-4c.18-.18.29-.43.29-.71 0-.28-.11-.53-.29-.71z' fill='%235C7080'/%3e%3c/svg%3e");
      content:"";
      display:block;
      height:16px;
      margin:0 5px;
      width:16px; }
    .bp3-breadcrumbs > li:last-of-type::after{
      display:none; }

.bp3-breadcrumb,
.bp3-breadcrumb-current,
.bp3-breadcrumbs-collapsed{
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  display:-webkit-inline-box;
  display:-ms-inline-flexbox;
  display:inline-flex;
  font-size:16px; }

.bp3-breadcrumb,
.bp3-breadcrumbs-collapsed{
  color:#5c7080; }

.bp3-breadcrumb:hover{
  text-decoration:none; }

.bp3-breadcrumb.bp3-disabled{
  color:rgba(92, 112, 128, 0.6);
  cursor:not-allowed; }

.bp3-breadcrumb .bp3-icon{
  margin-right:5px; }

.bp3-breadcrumb-current{
  color:inherit;
  font-weight:600; }
  .bp3-breadcrumb-current .bp3-input{
    font-size:inherit;
    font-weight:inherit;
    vertical-align:baseline; }

.bp3-breadcrumbs-collapsed{
  background:#ced9e0;
  border:none;
  border-radius:3px;
  cursor:pointer;
  margin-right:2px;
  padding:1px 5px;
  vertical-align:text-bottom; }
  .bp3-breadcrumbs-collapsed::before{
    background:url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'%3e%3cg fill='%235C7080'%3e%3ccircle cx='2' cy='8.03' r='2'/%3e%3ccircle cx='14' cy='8.03' r='2'/%3e%3ccircle cx='8' cy='8.03' r='2'/%3e%3c/g%3e%3c/svg%3e") center no-repeat;
    content:"";
    display:block;
    height:16px;
    width:16px; }
  .bp3-breadcrumbs-collapsed:hover{
    background:#bfccd6;
    color:#182026;
    text-decoration:none; }

.bp3-dark .bp3-breadcrumb,
.bp3-dark .bp3-breadcrumbs-collapsed{
  color:#a7b6c2; }

.bp3-dark .bp3-breadcrumbs > li::after{
  color:#a7b6c2; }

.bp3-dark .bp3-breadcrumb.bp3-disabled{
  color:rgba(167, 182, 194, 0.6); }

.bp3-dark .bp3-breadcrumb-current{
  color:#f5f8fa; }

.bp3-dark .bp3-breadcrumbs-collapsed{
  background:rgba(16, 22, 26, 0.4); }
  .bp3-dark .bp3-breadcrumbs-collapsed:hover{
    background:rgba(16, 22, 26, 0.6);
    color:#f5f8fa; }
.bp3-button{
  display:-webkit-inline-box;
  display:-ms-inline-flexbox;
  display:inline-flex;
  -webkit-box-orient:horizontal;
  -webkit-box-direction:normal;
      -ms-flex-direction:row;
          flex-direction:row;
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  border:none;
  border-radius:3px;
  cursor:pointer;
  font-size:14px;
  -webkit-box-pack:center;
      -ms-flex-pack:center;
          justify-content:center;
  padding:5px 10px;
  text-align:left;
  vertical-align:middle;
  min-height:30px;
  min-width:30px; }
  .bp3-button > *{
    -webkit-box-flex:0;
        -ms-flex-positive:0;
            flex-grow:0;
    -ms-flex-negative:0;
        flex-shrink:0; }
  .bp3-button > .bp3-fill{
    -webkit-box-flex:1;
        -ms-flex-positive:1;
            flex-grow:1;
    -ms-flex-negative:1;
        flex-shrink:1; }
  .bp3-button::before,
  .bp3-button > *{
    margin-right:7px; }
  .bp3-button:empty::before,
  .bp3-button > :last-child{
    margin-right:0; }
  .bp3-button:empty{
    padding:0 !important; }
  .bp3-button:disabled, .bp3-button.bp3-disabled{
    cursor:not-allowed; }
  .bp3-button.bp3-fill{
    display:-webkit-box;
    display:-ms-flexbox;
    display:flex;
    width:100%; }
  .bp3-button.bp3-align-right,
  .bp3-align-right .bp3-button{
    text-align:right; }
  .bp3-button.bp3-align-left,
  .bp3-align-left .bp3-button{
    text-align:left; }
  .bp3-button:not([class*="bp3-intent-"]){
    background-color:#f5f8fa;
    background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.8)), to(rgba(255, 255, 255, 0)));
    background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.8), rgba(255, 255, 255, 0));
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
    color:#182026; }
    .bp3-button:not([class*="bp3-intent-"]):hover{
      background-clip:padding-box;
      background-color:#ebf1f5;
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1); }
    .bp3-button:not([class*="bp3-intent-"]):active, .bp3-button:not([class*="bp3-intent-"]).bp3-active{
      background-color:#d8e1e8;
      background-image:none;
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 2px rgba(16, 22, 26, 0.2);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
    .bp3-button:not([class*="bp3-intent-"]):disabled, .bp3-button:not([class*="bp3-intent-"]).bp3-disabled{
      background-color:rgba(206, 217, 224, 0.5);
      background-image:none;
      -webkit-box-shadow:none;
              box-shadow:none;
      color:rgba(92, 112, 128, 0.6);
      cursor:not-allowed;
      outline:none; }
      .bp3-button:not([class*="bp3-intent-"]):disabled.bp3-active, .bp3-button:not([class*="bp3-intent-"]):disabled.bp3-active:hover, .bp3-button:not([class*="bp3-intent-"]).bp3-disabled.bp3-active, .bp3-button:not([class*="bp3-intent-"]).bp3-disabled.bp3-active:hover{
        background:rgba(206, 217, 224, 0.7); }
  .bp3-button.bp3-intent-primary{
    background-color:#137cbd;
    background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.1)), to(rgba(255, 255, 255, 0)));
    background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0));
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
    color:#ffffff; }
    .bp3-button.bp3-intent-primary:hover, .bp3-button.bp3-intent-primary:active, .bp3-button.bp3-intent-primary.bp3-active{
      color:#ffffff; }
    .bp3-button.bp3-intent-primary:hover{
      background-color:#106ba3;
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2); }
    .bp3-button.bp3-intent-primary:active, .bp3-button.bp3-intent-primary.bp3-active{
      background-color:#0e5a8a;
      background-image:none;
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
    .bp3-button.bp3-intent-primary:disabled, .bp3-button.bp3-intent-primary.bp3-disabled{
      background-color:rgba(19, 124, 189, 0.5);
      background-image:none;
      border-color:transparent;
      -webkit-box-shadow:none;
              box-shadow:none;
      color:rgba(255, 255, 255, 0.6); }
  .bp3-button.bp3-intent-success{
    background-color:#0f9960;
    background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.1)), to(rgba(255, 255, 255, 0)));
    background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0));
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
    color:#ffffff; }
    .bp3-button.bp3-intent-success:hover, .bp3-button.bp3-intent-success:active, .bp3-button.bp3-intent-success.bp3-active{
      color:#ffffff; }
    .bp3-button.bp3-intent-success:hover{
      background-color:#0d8050;
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2); }
    .bp3-button.bp3-intent-success:active, .bp3-button.bp3-intent-success.bp3-active{
      background-color:#0a6640;
      background-image:none;
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
    .bp3-button.bp3-intent-success:disabled, .bp3-button.bp3-intent-success.bp3-disabled{
      background-color:rgba(15, 153, 96, 0.5);
      background-image:none;
      border-color:transparent;
      -webkit-box-shadow:none;
              box-shadow:none;
      color:rgba(255, 255, 255, 0.6); }
  .bp3-button.bp3-intent-warning{
    background-color:#d9822b;
    background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.1)), to(rgba(255, 255, 255, 0)));
    background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0));
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
    color:#ffffff; }
    .bp3-button.bp3-intent-warning:hover, .bp3-button.bp3-intent-warning:active, .bp3-button.bp3-intent-warning.bp3-active{
      color:#ffffff; }
    .bp3-button.bp3-intent-warning:hover{
      background-color:#bf7326;
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2); }
    .bp3-button.bp3-intent-warning:active, .bp3-button.bp3-intent-warning.bp3-active{
      background-color:#a66321;
      background-image:none;
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
    .bp3-button.bp3-intent-warning:disabled, .bp3-button.bp3-intent-warning.bp3-disabled{
      background-color:rgba(217, 130, 43, 0.5);
      background-image:none;
      border-color:transparent;
      -webkit-box-shadow:none;
              box-shadow:none;
      color:rgba(255, 255, 255, 0.6); }
  .bp3-button.bp3-intent-danger{
    background-color:#db3737;
    background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.1)), to(rgba(255, 255, 255, 0)));
    background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0));
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
    color:#ffffff; }
    .bp3-button.bp3-intent-danger:hover, .bp3-button.bp3-intent-danger:active, .bp3-button.bp3-intent-danger.bp3-active{
      color:#ffffff; }
    .bp3-button.bp3-intent-danger:hover{
      background-color:#c23030;
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2); }
    .bp3-button.bp3-intent-danger:active, .bp3-button.bp3-intent-danger.bp3-active{
      background-color:#a82a2a;
      background-image:none;
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
    .bp3-button.bp3-intent-danger:disabled, .bp3-button.bp3-intent-danger.bp3-disabled{
      background-color:rgba(219, 55, 55, 0.5);
      background-image:none;
      border-color:transparent;
      -webkit-box-shadow:none;
              box-shadow:none;
      color:rgba(255, 255, 255, 0.6); }
  .bp3-button[class*="bp3-intent-"] .bp3-button-spinner .bp3-spinner-head{
    stroke:#ffffff; }
  .bp3-button.bp3-large,
  .bp3-large .bp3-button{
    min-height:40px;
    min-width:40px;
    font-size:16px;
    padding:5px 15px; }
    .bp3-button.bp3-large::before,
    .bp3-button.bp3-large > *,
    .bp3-large .bp3-button::before,
    .bp3-large .bp3-button > *{
      margin-right:10px; }
    .bp3-button.bp3-large:empty::before,
    .bp3-button.bp3-large > :last-child,
    .bp3-large .bp3-button:empty::before,
    .bp3-large .bp3-button > :last-child{
      margin-right:0; }
  .bp3-button.bp3-small,
  .bp3-small .bp3-button{
    min-height:24px;
    min-width:24px;
    padding:0 7px; }
  .bp3-button.bp3-loading{
    position:relative; }
    .bp3-button.bp3-loading[class*="bp3-icon-"]::before{
      visibility:hidden; }
    .bp3-button.bp3-loading .bp3-button-spinner{
      margin:0;
      position:absolute; }
    .bp3-button.bp3-loading > :not(.bp3-button-spinner){
      visibility:hidden; }
  .bp3-button[class*="bp3-icon-"]::before{
    font-family:"Icons16", sans-serif;
    font-size:16px;
    font-style:normal;
    font-weight:400;
    line-height:1;
    -moz-osx-font-smoothing:grayscale;
    -webkit-font-smoothing:antialiased;
    color:#5c7080; }
  .bp3-button .bp3-icon, .bp3-button .bp3-icon-standard, .bp3-button .bp3-icon-large{
    color:#5c7080; }
    .bp3-button .bp3-icon.bp3-align-right, .bp3-button .bp3-icon-standard.bp3-align-right, .bp3-button .bp3-icon-large.bp3-align-right{
      margin-left:7px; }
  .bp3-button .bp3-icon:first-child:last-child,
  .bp3-button .bp3-spinner + .bp3-icon:last-child{
    margin:0 -7px; }
  .bp3-dark .bp3-button:not([class*="bp3-intent-"]){
    background-color:#394b59;
    background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.05)), to(rgba(255, 255, 255, 0)));
    background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0));
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
    color:#f5f8fa; }
    .bp3-dark .bp3-button:not([class*="bp3-intent-"]):hover, .bp3-dark .bp3-button:not([class*="bp3-intent-"]):active, .bp3-dark .bp3-button:not([class*="bp3-intent-"]).bp3-active{
      color:#f5f8fa; }
    .bp3-dark .bp3-button:not([class*="bp3-intent-"]):hover{
      background-color:#30404d;
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4); }
    .bp3-dark .bp3-button:not([class*="bp3-intent-"]):active, .bp3-dark .bp3-button:not([class*="bp3-intent-"]).bp3-active{
      background-color:#202b33;
      background-image:none;
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.6), inset 0 1px 2px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.6), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
    .bp3-dark .bp3-button:not([class*="bp3-intent-"]):disabled, .bp3-dark .bp3-button:not([class*="bp3-intent-"]).bp3-disabled{
      background-color:rgba(57, 75, 89, 0.5);
      background-image:none;
      -webkit-box-shadow:none;
              box-shadow:none;
      color:rgba(167, 182, 194, 0.6); }
      .bp3-dark .bp3-button:not([class*="bp3-intent-"]):disabled.bp3-active, .bp3-dark .bp3-button:not([class*="bp3-intent-"]).bp3-disabled.bp3-active{
        background:rgba(57, 75, 89, 0.7); }
    .bp3-dark .bp3-button:not([class*="bp3-intent-"]) .bp3-button-spinner .bp3-spinner-head{
      background:rgba(16, 22, 26, 0.5);
      stroke:#8a9ba8; }
    .bp3-dark .bp3-button:not([class*="bp3-intent-"])[class*="bp3-icon-"]::before{
      color:#a7b6c2; }
    .bp3-dark .bp3-button:not([class*="bp3-intent-"]) .bp3-icon, .bp3-dark .bp3-button:not([class*="bp3-intent-"]) .bp3-icon-standard, .bp3-dark .bp3-button:not([class*="bp3-intent-"]) .bp3-icon-large{
      color:#a7b6c2; }
  .bp3-dark .bp3-button[class*="bp3-intent-"]{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4); }
    .bp3-dark .bp3-button[class*="bp3-intent-"]:hover{
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4); }
    .bp3-dark .bp3-button[class*="bp3-intent-"]:active, .bp3-dark .bp3-button[class*="bp3-intent-"].bp3-active{
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
    .bp3-dark .bp3-button[class*="bp3-intent-"]:disabled, .bp3-dark .bp3-button[class*="bp3-intent-"].bp3-disabled{
      background-image:none;
      -webkit-box-shadow:none;
              box-shadow:none;
      color:rgba(255, 255, 255, 0.3); }
    .bp3-dark .bp3-button[class*="bp3-intent-"] .bp3-button-spinner .bp3-spinner-head{
      stroke:#8a9ba8; }
  .bp3-button:disabled::before,
  .bp3-button:disabled .bp3-icon, .bp3-button:disabled .bp3-icon-standard, .bp3-button:disabled .bp3-icon-large, .bp3-button.bp3-disabled::before,
  .bp3-button.bp3-disabled .bp3-icon, .bp3-button.bp3-disabled .bp3-icon-standard, .bp3-button.bp3-disabled .bp3-icon-large, .bp3-button[class*="bp3-intent-"]::before,
  .bp3-button[class*="bp3-intent-"] .bp3-icon, .bp3-button[class*="bp3-intent-"] .bp3-icon-standard, .bp3-button[class*="bp3-intent-"] .bp3-icon-large{
    color:inherit !important; }
  .bp3-button.bp3-minimal{
    background:none;
    -webkit-box-shadow:none;
            box-shadow:none; }
    .bp3-button.bp3-minimal:hover{
      background:rgba(167, 182, 194, 0.3);
      -webkit-box-shadow:none;
              box-shadow:none;
      color:#182026;
      text-decoration:none; }
    .bp3-button.bp3-minimal:active, .bp3-button.bp3-minimal.bp3-active{
      background:rgba(115, 134, 148, 0.3);
      -webkit-box-shadow:none;
              box-shadow:none;
      color:#182026; }
    .bp3-button.bp3-minimal:disabled, .bp3-button.bp3-minimal:disabled:hover, .bp3-button.bp3-minimal.bp3-disabled, .bp3-button.bp3-minimal.bp3-disabled:hover{
      background:none;
      color:rgba(92, 112, 128, 0.6);
      cursor:not-allowed; }
      .bp3-button.bp3-minimal:disabled.bp3-active, .bp3-button.bp3-minimal:disabled:hover.bp3-active, .bp3-button.bp3-minimal.bp3-disabled.bp3-active, .bp3-button.bp3-minimal.bp3-disabled:hover.bp3-active{
        background:rgba(115, 134, 148, 0.3); }
    .bp3-dark .bp3-button.bp3-minimal{
      background:none;
      -webkit-box-shadow:none;
              box-shadow:none;
      color:inherit; }
      .bp3-dark .bp3-button.bp3-minimal:hover, .bp3-dark .bp3-button.bp3-minimal:active, .bp3-dark .bp3-button.bp3-minimal.bp3-active{
        background:none;
        -webkit-box-shadow:none;
                box-shadow:none; }
      .bp3-dark .bp3-button.bp3-minimal:hover{
        background:rgba(138, 155, 168, 0.15); }
      .bp3-dark .bp3-button.bp3-minimal:active, .bp3-dark .bp3-button.bp3-minimal.bp3-active{
        background:rgba(138, 155, 168, 0.3);
        color:#f5f8fa; }
      .bp3-dark .bp3-button.bp3-minimal:disabled, .bp3-dark .bp3-button.bp3-minimal:disabled:hover, .bp3-dark .bp3-button.bp3-minimal.bp3-disabled, .bp3-dark .bp3-button.bp3-minimal.bp3-disabled:hover{
        background:none;
        color:rgba(167, 182, 194, 0.6);
        cursor:not-allowed; }
        .bp3-dark .bp3-button.bp3-minimal:disabled.bp3-active, .bp3-dark .bp3-button.bp3-minimal:disabled:hover.bp3-active, .bp3-dark .bp3-button.bp3-minimal.bp3-disabled.bp3-active, .bp3-dark .bp3-button.bp3-minimal.bp3-disabled:hover.bp3-active{
          background:rgba(138, 155, 168, 0.3); }
    .bp3-button.bp3-minimal.bp3-intent-primary{
      color:#106ba3; }
      .bp3-button.bp3-minimal.bp3-intent-primary:hover, .bp3-button.bp3-minimal.bp3-intent-primary:active, .bp3-button.bp3-minimal.bp3-intent-primary.bp3-active{
        background:none;
        -webkit-box-shadow:none;
                box-shadow:none;
        color:#106ba3; }
      .bp3-button.bp3-minimal.bp3-intent-primary:hover{
        background:rgba(19, 124, 189, 0.15);
        color:#106ba3; }
      .bp3-button.bp3-minimal.bp3-intent-primary:active, .bp3-button.bp3-minimal.bp3-intent-primary.bp3-active{
        background:rgba(19, 124, 189, 0.3);
        color:#106ba3; }
      .bp3-button.bp3-minimal.bp3-intent-primary:disabled, .bp3-button.bp3-minimal.bp3-intent-primary.bp3-disabled{
        background:none;
        color:rgba(16, 107, 163, 0.5); }
        .bp3-button.bp3-minimal.bp3-intent-primary:disabled.bp3-active, .bp3-button.bp3-minimal.bp3-intent-primary.bp3-disabled.bp3-active{
          background:rgba(19, 124, 189, 0.3); }
      .bp3-button.bp3-minimal.bp3-intent-primary .bp3-button-spinner .bp3-spinner-head{
        stroke:#106ba3; }
      .bp3-dark .bp3-button.bp3-minimal.bp3-intent-primary{
        color:#48aff0; }
        .bp3-dark .bp3-button.bp3-minimal.bp3-intent-primary:hover{
          background:rgba(19, 124, 189, 0.2);
          color:#48aff0; }
        .bp3-dark .bp3-button.bp3-minimal.bp3-intent-primary:active, .bp3-dark .bp3-button.bp3-minimal.bp3-intent-primary.bp3-active{
          background:rgba(19, 124, 189, 0.3);
          color:#48aff0; }
        .bp3-dark .bp3-button.bp3-minimal.bp3-intent-primary:disabled, .bp3-dark .bp3-button.bp3-minimal.bp3-intent-primary.bp3-disabled{
          background:none;
          color:rgba(72, 175, 240, 0.5); }
          .bp3-dark .bp3-button.bp3-minimal.bp3-intent-primary:disabled.bp3-active, .bp3-dark .bp3-button.bp3-minimal.bp3-intent-primary.bp3-disabled.bp3-active{
            background:rgba(19, 124, 189, 0.3); }
    .bp3-button.bp3-minimal.bp3-intent-success{
      color:#0d8050; }
      .bp3-button.bp3-minimal.bp3-intent-success:hover, .bp3-button.bp3-minimal.bp3-intent-success:active, .bp3-button.bp3-minimal.bp3-intent-success.bp3-active{
        background:none;
        -webkit-box-shadow:none;
                box-shadow:none;
        color:#0d8050; }
      .bp3-button.bp3-minimal.bp3-intent-success:hover{
        background:rgba(15, 153, 96, 0.15);
        color:#0d8050; }
      .bp3-button.bp3-minimal.bp3-intent-success:active, .bp3-button.bp3-minimal.bp3-intent-success.bp3-active{
        background:rgba(15, 153, 96, 0.3);
        color:#0d8050; }
      .bp3-button.bp3-minimal.bp3-intent-success:disabled, .bp3-button.bp3-minimal.bp3-intent-success.bp3-disabled{
        background:none;
        color:rgba(13, 128, 80, 0.5); }
        .bp3-button.bp3-minimal.bp3-intent-success:disabled.bp3-active, .bp3-button.bp3-minimal.bp3-intent-success.bp3-disabled.bp3-active{
          background:rgba(15, 153, 96, 0.3); }
      .bp3-button.bp3-minimal.bp3-intent-success .bp3-button-spinner .bp3-spinner-head{
        stroke:#0d8050; }
      .bp3-dark .bp3-button.bp3-minimal.bp3-intent-success{
        color:#3dcc91; }
        .bp3-dark .bp3-button.bp3-minimal.bp3-intent-success:hover{
          background:rgba(15, 153, 96, 0.2);
          color:#3dcc91; }
        .bp3-dark .bp3-button.bp3-minimal.bp3-intent-success:active, .bp3-dark .bp3-button.bp3-minimal.bp3-intent-success.bp3-active{
          background:rgba(15, 153, 96, 0.3);
          color:#3dcc91; }
        .bp3-dark .bp3-button.bp3-minimal.bp3-intent-success:disabled, .bp3-dark .bp3-button.bp3-minimal.bp3-intent-success.bp3-disabled{
          background:none;
          color:rgba(61, 204, 145, 0.5); }
          .bp3-dark .bp3-button.bp3-minimal.bp3-intent-success:disabled.bp3-active, .bp3-dark .bp3-button.bp3-minimal.bp3-intent-success.bp3-disabled.bp3-active{
            background:rgba(15, 153, 96, 0.3); }
    .bp3-button.bp3-minimal.bp3-intent-warning{
      color:#bf7326; }
      .bp3-button.bp3-minimal.bp3-intent-warning:hover, .bp3-button.bp3-minimal.bp3-intent-warning:active, .bp3-button.bp3-minimal.bp3-intent-warning.bp3-active{
        background:none;
        -webkit-box-shadow:none;
                box-shadow:none;
        color:#bf7326; }
      .bp3-button.bp3-minimal.bp3-intent-warning:hover{
        background:rgba(217, 130, 43, 0.15);
        color:#bf7326; }
      .bp3-button.bp3-minimal.bp3-intent-warning:active, .bp3-button.bp3-minimal.bp3-intent-warning.bp3-active{
        background:rgba(217, 130, 43, 0.3);
        color:#bf7326; }
      .bp3-button.bp3-minimal.bp3-intent-warning:disabled, .bp3-button.bp3-minimal.bp3-intent-warning.bp3-disabled{
        background:none;
        color:rgba(191, 115, 38, 0.5); }
        .bp3-button.bp3-minimal.bp3-intent-warning:disabled.bp3-active, .bp3-button.bp3-minimal.bp3-intent-warning.bp3-disabled.bp3-active{
          background:rgba(217, 130, 43, 0.3); }
      .bp3-button.bp3-minimal.bp3-intent-warning .bp3-button-spinner .bp3-spinner-head{
        stroke:#bf7326; }
      .bp3-dark .bp3-button.bp3-minimal.bp3-intent-warning{
        color:#ffb366; }
        .bp3-dark .bp3-button.bp3-minimal.bp3-intent-warning:hover{
          background:rgba(217, 130, 43, 0.2);
          color:#ffb366; }
        .bp3-dark .bp3-button.bp3-minimal.bp3-intent-warning:active, .bp3-dark .bp3-button.bp3-minimal.bp3-intent-warning.bp3-active{
          background:rgba(217, 130, 43, 0.3);
          color:#ffb366; }
        .bp3-dark .bp3-button.bp3-minimal.bp3-intent-warning:disabled, .bp3-dark .bp3-button.bp3-minimal.bp3-intent-warning.bp3-disabled{
          background:none;
          color:rgba(255, 179, 102, 0.5); }
          .bp3-dark .bp3-button.bp3-minimal.bp3-intent-warning:disabled.bp3-active, .bp3-dark .bp3-button.bp3-minimal.bp3-intent-warning.bp3-disabled.bp3-active{
            background:rgba(217, 130, 43, 0.3); }
    .bp3-button.bp3-minimal.bp3-intent-danger{
      color:#c23030; }
      .bp3-button.bp3-minimal.bp3-intent-danger:hover, .bp3-button.bp3-minimal.bp3-intent-danger:active, .bp3-button.bp3-minimal.bp3-intent-danger.bp3-active{
        background:none;
        -webkit-box-shadow:none;
                box-shadow:none;
        color:#c23030; }
      .bp3-button.bp3-minimal.bp3-intent-danger:hover{
        background:rgba(219, 55, 55, 0.15);
        color:#c23030; }
      .bp3-button.bp3-minimal.bp3-intent-danger:active, .bp3-button.bp3-minimal.bp3-intent-danger.bp3-active{
        background:rgba(219, 55, 55, 0.3);
        color:#c23030; }
      .bp3-button.bp3-minimal.bp3-intent-danger:disabled, .bp3-button.bp3-minimal.bp3-intent-danger.bp3-disabled{
        background:none;
        color:rgba(194, 48, 48, 0.5); }
        .bp3-button.bp3-minimal.bp3-intent-danger:disabled.bp3-active, .bp3-button.bp3-minimal.bp3-intent-danger.bp3-disabled.bp3-active{
          background:rgba(219, 55, 55, 0.3); }
      .bp3-button.bp3-minimal.bp3-intent-danger .bp3-button-spinner .bp3-spinner-head{
        stroke:#c23030; }
      .bp3-dark .bp3-button.bp3-minimal.bp3-intent-danger{
        color:#ff7373; }
        .bp3-dark .bp3-button.bp3-minimal.bp3-intent-danger:hover{
          background:rgba(219, 55, 55, 0.2);
          color:#ff7373; }
        .bp3-dark .bp3-button.bp3-minimal.bp3-intent-danger:active, .bp3-dark .bp3-button.bp3-minimal.bp3-intent-danger.bp3-active{
          background:rgba(219, 55, 55, 0.3);
          color:#ff7373; }
        .bp3-dark .bp3-button.bp3-minimal.bp3-intent-danger:disabled, .bp3-dark .bp3-button.bp3-minimal.bp3-intent-danger.bp3-disabled{
          background:none;
          color:rgba(255, 115, 115, 0.5); }
          .bp3-dark .bp3-button.bp3-minimal.bp3-intent-danger:disabled.bp3-active, .bp3-dark .bp3-button.bp3-minimal.bp3-intent-danger.bp3-disabled.bp3-active{
            background:rgba(219, 55, 55, 0.3); }
  .bp3-button.bp3-outlined{
    background:none;
    -webkit-box-shadow:none;
            box-shadow:none;
    border:1px solid rgba(24, 32, 38, 0.2);
    -webkit-box-sizing:border-box;
            box-sizing:border-box; }
    .bp3-button.bp3-outlined:hover{
      background:rgba(167, 182, 194, 0.3);
      -webkit-box-shadow:none;
              box-shadow:none;
      color:#182026;
      text-decoration:none; }
    .bp3-button.bp3-outlined:active, .bp3-button.bp3-outlined.bp3-active{
      background:rgba(115, 134, 148, 0.3);
      -webkit-box-shadow:none;
              box-shadow:none;
      color:#182026; }
    .bp3-button.bp3-outlined:disabled, .bp3-button.bp3-outlined:disabled:hover, .bp3-button.bp3-outlined.bp3-disabled, .bp3-button.bp3-outlined.bp3-disabled:hover{
      background:none;
      color:rgba(92, 112, 128, 0.6);
      cursor:not-allowed; }
      .bp3-button.bp3-outlined:disabled.bp3-active, .bp3-button.bp3-outlined:disabled:hover.bp3-active, .bp3-button.bp3-outlined.bp3-disabled.bp3-active, .bp3-button.bp3-outlined.bp3-disabled:hover.bp3-active{
        background:rgba(115, 134, 148, 0.3); }
    .bp3-dark .bp3-button.bp3-outlined{
      background:none;
      -webkit-box-shadow:none;
              box-shadow:none;
      color:inherit; }
      .bp3-dark .bp3-button.bp3-outlined:hover, .bp3-dark .bp3-button.bp3-outlined:active, .bp3-dark .bp3-button.bp3-outlined.bp3-active{
        background:none;
        -webkit-box-shadow:none;
                box-shadow:none; }
      .bp3-dark .bp3-button.bp3-outlined:hover{
        background:rgba(138, 155, 168, 0.15); }
      .bp3-dark .bp3-button.bp3-outlined:active, .bp3-dark .bp3-button.bp3-outlined.bp3-active{
        background:rgba(138, 155, 168, 0.3);
        color:#f5f8fa; }
      .bp3-dark .bp3-button.bp3-outlined:disabled, .bp3-dark .bp3-button.bp3-outlined:disabled:hover, .bp3-dark .bp3-button.bp3-outlined.bp3-disabled, .bp3-dark .bp3-button.bp3-outlined.bp3-disabled:hover{
        background:none;
        color:rgba(167, 182, 194, 0.6);
        cursor:not-allowed; }
        .bp3-dark .bp3-button.bp3-outlined:disabled.bp3-active, .bp3-dark .bp3-button.bp3-outlined:disabled:hover.bp3-active, .bp3-dark .bp3-button.bp3-outlined.bp3-disabled.bp3-active, .bp3-dark .bp3-button.bp3-outlined.bp3-disabled:hover.bp3-active{
          background:rgba(138, 155, 168, 0.3); }
    .bp3-button.bp3-outlined.bp3-intent-primary{
      color:#106ba3; }
      .bp3-button.bp3-outlined.bp3-intent-primary:hover, .bp3-button.bp3-outlined.bp3-intent-primary:active, .bp3-button.bp3-outlined.bp3-intent-primary.bp3-active{
        background:none;
        -webkit-box-shadow:none;
                box-shadow:none;
        color:#106ba3; }
      .bp3-button.bp3-outlined.bp3-intent-primary:hover{
        background:rgba(19, 124, 189, 0.15);
        color:#106ba3; }
      .bp3-button.bp3-outlined.bp3-intent-primary:active, .bp3-button.bp3-outlined.bp3-intent-primary.bp3-active{
        background:rgba(19, 124, 189, 0.3);
        color:#106ba3; }
      .bp3-button.bp3-outlined.bp3-intent-primary:disabled, .bp3-button.bp3-outlined.bp3-intent-primary.bp3-disabled{
        background:none;
        color:rgba(16, 107, 163, 0.5); }
        .bp3-button.bp3-outlined.bp3-intent-primary:disabled.bp3-active, .bp3-button.bp3-outlined.bp3-intent-primary.bp3-disabled.bp3-active{
          background:rgba(19, 124, 189, 0.3); }
      .bp3-button.bp3-outlined.bp3-intent-primary .bp3-button-spinner .bp3-spinner-head{
        stroke:#106ba3; }
      .bp3-dark .bp3-button.bp3-outlined.bp3-intent-primary{
        color:#48aff0; }
        .bp3-dark .bp3-button.bp3-outlined.bp3-intent-primary:hover{
          background:rgba(19, 124, 189, 0.2);
          color:#48aff0; }
        .bp3-dark .bp3-button.bp3-outlined.bp3-intent-primary:active, .bp3-dark .bp3-button.bp3-outlined.bp3-intent-primary.bp3-active{
          background:rgba(19, 124, 189, 0.3);
          color:#48aff0; }
        .bp3-dark .bp3-button.bp3-outlined.bp3-intent-primary:disabled, .bp3-dark .bp3-button.bp3-outlined.bp3-intent-primary.bp3-disabled{
          background:none;
          color:rgba(72, 175, 240, 0.5); }
          .bp3-dark .bp3-button.bp3-outlined.bp3-intent-primary:disabled.bp3-active, .bp3-dark .bp3-button.bp3-outlined.bp3-intent-primary.bp3-disabled.bp3-active{
            background:rgba(19, 124, 189, 0.3); }
    .bp3-button.bp3-outlined.bp3-intent-success{
      color:#0d8050; }
      .bp3-button.bp3-outlined.bp3-intent-success:hover, .bp3-button.bp3-outlined.bp3-intent-success:active, .bp3-button.bp3-outlined.bp3-intent-success.bp3-active{
        background:none;
        -webkit-box-shadow:none;
                box-shadow:none;
        color:#0d8050; }
      .bp3-button.bp3-outlined.bp3-intent-success:hover{
        background:rgba(15, 153, 96, 0.15);
        color:#0d8050; }
      .bp3-button.bp3-outlined.bp3-intent-success:active, .bp3-button.bp3-outlined.bp3-intent-success.bp3-active{
        background:rgba(15, 153, 96, 0.3);
        color:#0d8050; }
      .bp3-button.bp3-outlined.bp3-intent-success:disabled, .bp3-button.bp3-outlined.bp3-intent-success.bp3-disabled{
        background:none;
        color:rgba(13, 128, 80, 0.5); }
        .bp3-button.bp3-outlined.bp3-intent-success:disabled.bp3-active, .bp3-button.bp3-outlined.bp3-intent-success.bp3-disabled.bp3-active{
          background:rgba(15, 153, 96, 0.3); }
      .bp3-button.bp3-outlined.bp3-intent-success .bp3-button-spinner .bp3-spinner-head{
        stroke:#0d8050; }
      .bp3-dark .bp3-button.bp3-outlined.bp3-intent-success{
        color:#3dcc91; }
        .bp3-dark .bp3-button.bp3-outlined.bp3-intent-success:hover{
          background:rgba(15, 153, 96, 0.2);
          color:#3dcc91; }
        .bp3-dark .bp3-button.bp3-outlined.bp3-intent-success:active, .bp3-dark .bp3-button.bp3-outlined.bp3-intent-success.bp3-active{
          background:rgba(15, 153, 96, 0.3);
          color:#3dcc91; }
        .bp3-dark .bp3-button.bp3-outlined.bp3-intent-success:disabled, .bp3-dark .bp3-button.bp3-outlined.bp3-intent-success.bp3-disabled{
          background:none;
          color:rgba(61, 204, 145, 0.5); }
          .bp3-dark .bp3-button.bp3-outlined.bp3-intent-success:disabled.bp3-active, .bp3-dark .bp3-button.bp3-outlined.bp3-intent-success.bp3-disabled.bp3-active{
            background:rgba(15, 153, 96, 0.3); }
    .bp3-button.bp3-outlined.bp3-intent-warning{
      color:#bf7326; }
      .bp3-button.bp3-outlined.bp3-intent-warning:hover, .bp3-button.bp3-outlined.bp3-intent-warning:active, .bp3-button.bp3-outlined.bp3-intent-warning.bp3-active{
        background:none;
        -webkit-box-shadow:none;
                box-shadow:none;
        color:#bf7326; }
      .bp3-button.bp3-outlined.bp3-intent-warning:hover{
        background:rgba(217, 130, 43, 0.15);
        color:#bf7326; }
      .bp3-button.bp3-outlined.bp3-intent-warning:active, .bp3-button.bp3-outlined.bp3-intent-warning.bp3-active{
        background:rgba(217, 130, 43, 0.3);
        color:#bf7326; }
      .bp3-button.bp3-outlined.bp3-intent-warning:disabled, .bp3-button.bp3-outlined.bp3-intent-warning.bp3-disabled{
        background:none;
        color:rgba(191, 115, 38, 0.5); }
        .bp3-button.bp3-outlined.bp3-intent-warning:disabled.bp3-active, .bp3-button.bp3-outlined.bp3-intent-warning.bp3-disabled.bp3-active{
          background:rgba(217, 130, 43, 0.3); }
      .bp3-button.bp3-outlined.bp3-intent-warning .bp3-button-spinner .bp3-spinner-head{
        stroke:#bf7326; }
      .bp3-dark .bp3-button.bp3-outlined.bp3-intent-warning{
        color:#ffb366; }
        .bp3-dark .bp3-button.bp3-outlined.bp3-intent-warning:hover{
          background:rgba(217, 130, 43, 0.2);
          color:#ffb366; }
        .bp3-dark .bp3-button.bp3-outlined.bp3-intent-warning:active, .bp3-dark .bp3-button.bp3-outlined.bp3-intent-warning.bp3-active{
          background:rgba(217, 130, 43, 0.3);
          color:#ffb366; }
        .bp3-dark .bp3-button.bp3-outlined.bp3-intent-warning:disabled, .bp3-dark .bp3-button.bp3-outlined.bp3-intent-warning.bp3-disabled{
          background:none;
          color:rgba(255, 179, 102, 0.5); }
          .bp3-dark .bp3-button.bp3-outlined.bp3-intent-warning:disabled.bp3-active, .bp3-dark .bp3-button.bp3-outlined.bp3-intent-warning.bp3-disabled.bp3-active{
            background:rgba(217, 130, 43, 0.3); }
    .bp3-button.bp3-outlined.bp3-intent-danger{
      color:#c23030; }
      .bp3-button.bp3-outlined.bp3-intent-danger:hover, .bp3-button.bp3-outlined.bp3-intent-danger:active, .bp3-button.bp3-outlined.bp3-intent-danger.bp3-active{
        background:none;
        -webkit-box-shadow:none;
                box-shadow:none;
        color:#c23030; }
      .bp3-button.bp3-outlined.bp3-intent-danger:hover{
        background:rgba(219, 55, 55, 0.15);
        color:#c23030; }
      .bp3-button.bp3-outlined.bp3-intent-danger:active, .bp3-button.bp3-outlined.bp3-intent-danger.bp3-active{
        background:rgba(219, 55, 55, 0.3);
        color:#c23030; }
      .bp3-button.bp3-outlined.bp3-intent-danger:disabled, .bp3-button.bp3-outlined.bp3-intent-danger.bp3-disabled{
        background:none;
        color:rgba(194, 48, 48, 0.5); }
        .bp3-button.bp3-outlined.bp3-intent-danger:disabled.bp3-active, .bp3-button.bp3-outlined.bp3-intent-danger.bp3-disabled.bp3-active{
          background:rgba(219, 55, 55, 0.3); }
      .bp3-button.bp3-outlined.bp3-intent-danger .bp3-button-spinner .bp3-spinner-head{
        stroke:#c23030; }
      .bp3-dark .bp3-button.bp3-outlined.bp3-intent-danger{
        color:#ff7373; }
        .bp3-dark .bp3-button.bp3-outlined.bp3-intent-danger:hover{
          background:rgba(219, 55, 55, 0.2);
          color:#ff7373; }
        .bp3-dark .bp3-button.bp3-outlined.bp3-intent-danger:active, .bp3-dark .bp3-button.bp3-outlined.bp3-intent-danger.bp3-active{
          background:rgba(219, 55, 55, 0.3);
          color:#ff7373; }
        .bp3-dark .bp3-button.bp3-outlined.bp3-intent-danger:disabled, .bp3-dark .bp3-button.bp3-outlined.bp3-intent-danger.bp3-disabled{
          background:none;
          color:rgba(255, 115, 115, 0.5); }
          .bp3-dark .bp3-button.bp3-outlined.bp3-intent-danger:disabled.bp3-active, .bp3-dark .bp3-button.bp3-outlined.bp3-intent-danger.bp3-disabled.bp3-active{
            background:rgba(219, 55, 55, 0.3); }
    .bp3-button.bp3-outlined:disabled, .bp3-button.bp3-outlined.bp3-disabled, .bp3-button.bp3-outlined:disabled:hover, .bp3-button.bp3-outlined.bp3-disabled:hover{
      border-color:rgba(92, 112, 128, 0.1); }
    .bp3-dark .bp3-button.bp3-outlined{
      border-color:rgba(255, 255, 255, 0.4); }
      .bp3-dark .bp3-button.bp3-outlined:disabled, .bp3-dark .bp3-button.bp3-outlined:disabled:hover, .bp3-dark .bp3-button.bp3-outlined.bp3-disabled, .bp3-dark .bp3-button.bp3-outlined.bp3-disabled:hover{
        border-color:rgba(255, 255, 255, 0.2); }
    .bp3-button.bp3-outlined.bp3-intent-primary{
      border-color:rgba(16, 107, 163, 0.6); }
      .bp3-button.bp3-outlined.bp3-intent-primary:disabled, .bp3-button.bp3-outlined.bp3-intent-primary.bp3-disabled{
        border-color:rgba(16, 107, 163, 0.2); }
      .bp3-dark .bp3-button.bp3-outlined.bp3-intent-primary{
        border-color:rgba(72, 175, 240, 0.6); }
        .bp3-dark .bp3-button.bp3-outlined.bp3-intent-primary:disabled, .bp3-dark .bp3-button.bp3-outlined.bp3-intent-primary.bp3-disabled{
          border-color:rgba(72, 175, 240, 0.2); }
    .bp3-button.bp3-outlined.bp3-intent-success{
      border-color:rgba(13, 128, 80, 0.6); }
      .bp3-button.bp3-outlined.bp3-intent-success:disabled, .bp3-button.bp3-outlined.bp3-intent-success.bp3-disabled{
        border-color:rgba(13, 128, 80, 0.2); }
      .bp3-dark .bp3-button.bp3-outlined.bp3-intent-success{
        border-color:rgba(61, 204, 145, 0.6); }
        .bp3-dark .bp3-button.bp3-outlined.bp3-intent-success:disabled, .bp3-dark .bp3-button.bp3-outlined.bp3-intent-success.bp3-disabled{
          border-color:rgba(61, 204, 145, 0.2); }
    .bp3-button.bp3-outlined.bp3-intent-warning{
      border-color:rgba(191, 115, 38, 0.6); }
      .bp3-button.bp3-outlined.bp3-intent-warning:disabled, .bp3-button.bp3-outlined.bp3-intent-warning.bp3-disabled{
        border-color:rgba(191, 115, 38, 0.2); }
      .bp3-dark .bp3-button.bp3-outlined.bp3-intent-warning{
        border-color:rgba(255, 179, 102, 0.6); }
        .bp3-dark .bp3-button.bp3-outlined.bp3-intent-warning:disabled, .bp3-dark .bp3-button.bp3-outlined.bp3-intent-warning.bp3-disabled{
          border-color:rgba(255, 179, 102, 0.2); }
    .bp3-button.bp3-outlined.bp3-intent-danger{
      border-color:rgba(194, 48, 48, 0.6); }
      .bp3-button.bp3-outlined.bp3-intent-danger:disabled, .bp3-button.bp3-outlined.bp3-intent-danger.bp3-disabled{
        border-color:rgba(194, 48, 48, 0.2); }
      .bp3-dark .bp3-button.bp3-outlined.bp3-intent-danger{
        border-color:rgba(255, 115, 115, 0.6); }
        .bp3-dark .bp3-button.bp3-outlined.bp3-intent-danger:disabled, .bp3-dark .bp3-button.bp3-outlined.bp3-intent-danger.bp3-disabled{
          border-color:rgba(255, 115, 115, 0.2); }

a.bp3-button{
  text-align:center;
  text-decoration:none;
  -webkit-transition:none;
  transition:none; }
  a.bp3-button, a.bp3-button:hover, a.bp3-button:active{
    color:#182026; }
  a.bp3-button.bp3-disabled{
    color:rgba(92, 112, 128, 0.6); }

.bp3-button-text{
  -webkit-box-flex:0;
      -ms-flex:0 1 auto;
          flex:0 1 auto; }

.bp3-button.bp3-align-left .bp3-button-text, .bp3-button.bp3-align-right .bp3-button-text,
.bp3-button-group.bp3-align-left .bp3-button-text,
.bp3-button-group.bp3-align-right .bp3-button-text{
  -webkit-box-flex:1;
      -ms-flex:1 1 auto;
          flex:1 1 auto; }
.bp3-button-group{
  display:-webkit-inline-box;
  display:-ms-inline-flexbox;
  display:inline-flex; }
  .bp3-button-group .bp3-button{
    -webkit-box-flex:0;
        -ms-flex:0 0 auto;
            flex:0 0 auto;
    position:relative;
    z-index:4; }
    .bp3-button-group .bp3-button:focus{
      z-index:5; }
    .bp3-button-group .bp3-button:hover{
      z-index:6; }
    .bp3-button-group .bp3-button:active, .bp3-button-group .bp3-button.bp3-active{
      z-index:7; }
    .bp3-button-group .bp3-button:disabled, .bp3-button-group .bp3-button.bp3-disabled{
      z-index:3; }
    .bp3-button-group .bp3-button[class*="bp3-intent-"]{
      z-index:9; }
      .bp3-button-group .bp3-button[class*="bp3-intent-"]:focus{
        z-index:10; }
      .bp3-button-group .bp3-button[class*="bp3-intent-"]:hover{
        z-index:11; }
      .bp3-button-group .bp3-button[class*="bp3-intent-"]:active, .bp3-button-group .bp3-button[class*="bp3-intent-"].bp3-active{
        z-index:12; }
      .bp3-button-group .bp3-button[class*="bp3-intent-"]:disabled, .bp3-button-group .bp3-button[class*="bp3-intent-"].bp3-disabled{
        z-index:8; }
  .bp3-button-group:not(.bp3-minimal) > .bp3-popover-wrapper:not(:first-child) .bp3-button,
  .bp3-button-group:not(.bp3-minimal) > .bp3-button:not(:first-child){
    border-bottom-left-radius:0;
    border-top-left-radius:0; }
  .bp3-button-group:not(.bp3-minimal) > .bp3-popover-wrapper:not(:last-child) .bp3-button,
  .bp3-button-group:not(.bp3-minimal) > .bp3-button:not(:last-child){
    border-bottom-right-radius:0;
    border-top-right-radius:0;
    margin-right:-1px; }
  .bp3-button-group.bp3-minimal .bp3-button{
    background:none;
    -webkit-box-shadow:none;
            box-shadow:none; }
    .bp3-button-group.bp3-minimal .bp3-button:hover{
      background:rgba(167, 182, 194, 0.3);
      -webkit-box-shadow:none;
              box-shadow:none;
      color:#182026;
      text-decoration:none; }
    .bp3-button-group.bp3-minimal .bp3-button:active, .bp3-button-group.bp3-minimal .bp3-button.bp3-active{
      background:rgba(115, 134, 148, 0.3);
      -webkit-box-shadow:none;
              box-shadow:none;
      color:#182026; }
    .bp3-button-group.bp3-minimal .bp3-button:disabled, .bp3-button-group.bp3-minimal .bp3-button:disabled:hover, .bp3-button-group.bp3-minimal .bp3-button.bp3-disabled, .bp3-button-group.bp3-minimal .bp3-button.bp3-disabled:hover{
      background:none;
      color:rgba(92, 112, 128, 0.6);
      cursor:not-allowed; }
      .bp3-button-group.bp3-minimal .bp3-button:disabled.bp3-active, .bp3-button-group.bp3-minimal .bp3-button:disabled:hover.bp3-active, .bp3-button-group.bp3-minimal .bp3-button.bp3-disabled.bp3-active, .bp3-button-group.bp3-minimal .bp3-button.bp3-disabled:hover.bp3-active{
        background:rgba(115, 134, 148, 0.3); }
    .bp3-dark .bp3-button-group.bp3-minimal .bp3-button{
      background:none;
      -webkit-box-shadow:none;
              box-shadow:none;
      color:inherit; }
      .bp3-dark .bp3-button-group.bp3-minimal .bp3-button:hover, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button:active, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-active{
        background:none;
        -webkit-box-shadow:none;
                box-shadow:none; }
      .bp3-dark .bp3-button-group.bp3-minimal .bp3-button:hover{
        background:rgba(138, 155, 168, 0.15); }
      .bp3-dark .bp3-button-group.bp3-minimal .bp3-button:active, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-active{
        background:rgba(138, 155, 168, 0.3);
        color:#f5f8fa; }
      .bp3-dark .bp3-button-group.bp3-minimal .bp3-button:disabled, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button:disabled:hover, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-disabled, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-disabled:hover{
        background:none;
        color:rgba(167, 182, 194, 0.6);
        cursor:not-allowed; }
        .bp3-dark .bp3-button-group.bp3-minimal .bp3-button:disabled.bp3-active, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button:disabled:hover.bp3-active, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-disabled.bp3-active, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-disabled:hover.bp3-active{
          background:rgba(138, 155, 168, 0.3); }
    .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary{
      color:#106ba3; }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary:hover, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary:active, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary.bp3-active{
        background:none;
        -webkit-box-shadow:none;
                box-shadow:none;
        color:#106ba3; }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary:hover{
        background:rgba(19, 124, 189, 0.15);
        color:#106ba3; }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary:active, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary.bp3-active{
        background:rgba(19, 124, 189, 0.3);
        color:#106ba3; }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary:disabled, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary.bp3-disabled{
        background:none;
        color:rgba(16, 107, 163, 0.5); }
        .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary:disabled.bp3-active, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary.bp3-disabled.bp3-active{
          background:rgba(19, 124, 189, 0.3); }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary .bp3-button-spinner .bp3-spinner-head{
        stroke:#106ba3; }
      .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary{
        color:#48aff0; }
        .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary:hover{
          background:rgba(19, 124, 189, 0.2);
          color:#48aff0; }
        .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary:active, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary.bp3-active{
          background:rgba(19, 124, 189, 0.3);
          color:#48aff0; }
        .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary:disabled, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary.bp3-disabled{
          background:none;
          color:rgba(72, 175, 240, 0.5); }
          .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary:disabled.bp3-active, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary.bp3-disabled.bp3-active{
            background:rgba(19, 124, 189, 0.3); }
    .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success{
      color:#0d8050; }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success:hover, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success:active, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success.bp3-active{
        background:none;
        -webkit-box-shadow:none;
                box-shadow:none;
        color:#0d8050; }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success:hover{
        background:rgba(15, 153, 96, 0.15);
        color:#0d8050; }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success:active, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success.bp3-active{
        background:rgba(15, 153, 96, 0.3);
        color:#0d8050; }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success:disabled, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success.bp3-disabled{
        background:none;
        color:rgba(13, 128, 80, 0.5); }
        .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success:disabled.bp3-active, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success.bp3-disabled.bp3-active{
          background:rgba(15, 153, 96, 0.3); }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success .bp3-button-spinner .bp3-spinner-head{
        stroke:#0d8050; }
      .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success{
        color:#3dcc91; }
        .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success:hover{
          background:rgba(15, 153, 96, 0.2);
          color:#3dcc91; }
        .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success:active, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success.bp3-active{
          background:rgba(15, 153, 96, 0.3);
          color:#3dcc91; }
        .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success:disabled, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success.bp3-disabled{
          background:none;
          color:rgba(61, 204, 145, 0.5); }
          .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success:disabled.bp3-active, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success.bp3-disabled.bp3-active{
            background:rgba(15, 153, 96, 0.3); }
    .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning{
      color:#bf7326; }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning:hover, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning:active, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning.bp3-active{
        background:none;
        -webkit-box-shadow:none;
                box-shadow:none;
        color:#bf7326; }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning:hover{
        background:rgba(217, 130, 43, 0.15);
        color:#bf7326; }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning:active, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning.bp3-active{
        background:rgba(217, 130, 43, 0.3);
        color:#bf7326; }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning:disabled, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning.bp3-disabled{
        background:none;
        color:rgba(191, 115, 38, 0.5); }
        .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning:disabled.bp3-active, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning.bp3-disabled.bp3-active{
          background:rgba(217, 130, 43, 0.3); }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning .bp3-button-spinner .bp3-spinner-head{
        stroke:#bf7326; }
      .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning{
        color:#ffb366; }
        .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning:hover{
          background:rgba(217, 130, 43, 0.2);
          color:#ffb366; }
        .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning:active, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning.bp3-active{
          background:rgba(217, 130, 43, 0.3);
          color:#ffb366; }
        .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning:disabled, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning.bp3-disabled{
          background:none;
          color:rgba(255, 179, 102, 0.5); }
          .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning:disabled.bp3-active, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning.bp3-disabled.bp3-active{
            background:rgba(217, 130, 43, 0.3); }
    .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger{
      color:#c23030; }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger:hover, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger:active, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger.bp3-active{
        background:none;
        -webkit-box-shadow:none;
                box-shadow:none;
        color:#c23030; }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger:hover{
        background:rgba(219, 55, 55, 0.15);
        color:#c23030; }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger:active, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger.bp3-active{
        background:rgba(219, 55, 55, 0.3);
        color:#c23030; }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger:disabled, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger.bp3-disabled{
        background:none;
        color:rgba(194, 48, 48, 0.5); }
        .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger:disabled.bp3-active, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger.bp3-disabled.bp3-active{
          background:rgba(219, 55, 55, 0.3); }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger .bp3-button-spinner .bp3-spinner-head{
        stroke:#c23030; }
      .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger{
        color:#ff7373; }
        .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger:hover{
          background:rgba(219, 55, 55, 0.2);
          color:#ff7373; }
        .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger:active, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger.bp3-active{
          background:rgba(219, 55, 55, 0.3);
          color:#ff7373; }
        .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger:disabled, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger.bp3-disabled{
          background:none;
          color:rgba(255, 115, 115, 0.5); }
          .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger:disabled.bp3-active, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger.bp3-disabled.bp3-active{
            background:rgba(219, 55, 55, 0.3); }
  .bp3-button-group .bp3-popover-wrapper,
  .bp3-button-group .bp3-popover-target{
    display:-webkit-box;
    display:-ms-flexbox;
    display:flex;
    -webkit-box-flex:1;
        -ms-flex:1 1 auto;
            flex:1 1 auto; }
  .bp3-button-group.bp3-fill{
    display:-webkit-box;
    display:-ms-flexbox;
    display:flex;
    width:100%; }
  .bp3-button-group .bp3-button.bp3-fill,
  .bp3-button-group.bp3-fill .bp3-button:not(.bp3-fixed){
    -webkit-box-flex:1;
        -ms-flex:1 1 auto;
            flex:1 1 auto; }
  .bp3-button-group.bp3-vertical{
    -webkit-box-align:stretch;
        -ms-flex-align:stretch;
            align-items:stretch;
    -webkit-box-orient:vertical;
    -webkit-box-direction:normal;
        -ms-flex-direction:column;
            flex-direction:column;
    vertical-align:top; }
    .bp3-button-group.bp3-vertical.bp3-fill{
      height:100%;
      width:unset; }
    .bp3-button-group.bp3-vertical .bp3-button{
      margin-right:0 !important;
      width:100%; }
    .bp3-button-group.bp3-vertical:not(.bp3-minimal) > .bp3-popover-wrapper:first-child .bp3-button,
    .bp3-button-group.bp3-vertical:not(.bp3-minimal) > .bp3-button:first-child{
      border-radius:3px 3px 0 0; }
    .bp3-button-group.bp3-vertical:not(.bp3-minimal) > .bp3-popover-wrapper:last-child .bp3-button,
    .bp3-button-group.bp3-vertical:not(.bp3-minimal) > .bp3-button:last-child{
      border-radius:0 0 3px 3px; }
    .bp3-button-group.bp3-vertical:not(.bp3-minimal) > .bp3-popover-wrapper:not(:last-child) .bp3-button,
    .bp3-button-group.bp3-vertical:not(.bp3-minimal) > .bp3-button:not(:last-child){
      margin-bottom:-1px; }
  .bp3-button-group.bp3-align-left .bp3-button{
    text-align:left; }
  .bp3-dark .bp3-button-group:not(.bp3-minimal) > .bp3-popover-wrapper:not(:last-child) .bp3-button,
  .bp3-dark .bp3-button-group:not(.bp3-minimal) > .bp3-button:not(:last-child){
    margin-right:1px; }
  .bp3-dark .bp3-button-group.bp3-vertical > .bp3-popover-wrapper:not(:last-child) .bp3-button,
  .bp3-dark .bp3-button-group.bp3-vertical > .bp3-button:not(:last-child){
    margin-bottom:1px; }
.bp3-callout{
  font-size:14px;
  line-height:1.5;
  background-color:rgba(138, 155, 168, 0.15);
  border-radius:3px;
  padding:10px 12px 9px;
  position:relative;
  width:100%; }
  .bp3-callout[class*="bp3-icon-"]{
    padding-left:40px; }
    .bp3-callout[class*="bp3-icon-"]::before{
      font-family:"Icons20", sans-serif;
      font-size:20px;
      font-style:normal;
      font-weight:400;
      line-height:1;
      -moz-osx-font-smoothing:grayscale;
      -webkit-font-smoothing:antialiased;
      color:#5c7080;
      left:10px;
      position:absolute;
      top:10px; }
  .bp3-callout.bp3-callout-icon{
    padding-left:40px; }
    .bp3-callout.bp3-callout-icon > .bp3-icon:first-child{
      color:#5c7080;
      left:10px;
      position:absolute;
      top:10px; }
  .bp3-callout .bp3-heading{
    line-height:20px;
    margin-bottom:5px;
    margin-top:0; }
    .bp3-callout .bp3-heading:last-child{
      margin-bottom:0; }
  .bp3-dark .bp3-callout{
    background-color:rgba(138, 155, 168, 0.2); }
    .bp3-dark .bp3-callout[class*="bp3-icon-"]::before{
      color:#a7b6c2; }
  .bp3-callout.bp3-intent-primary{
    background-color:rgba(19, 124, 189, 0.15); }
    .bp3-callout.bp3-intent-primary[class*="bp3-icon-"]::before,
    .bp3-callout.bp3-intent-primary > .bp3-icon:first-child,
    .bp3-callout.bp3-intent-primary .bp3-heading{
      color:#106ba3; }
    .bp3-dark .bp3-callout.bp3-intent-primary{
      background-color:rgba(19, 124, 189, 0.25); }
      .bp3-dark .bp3-callout.bp3-intent-primary[class*="bp3-icon-"]::before,
      .bp3-dark .bp3-callout.bp3-intent-primary > .bp3-icon:first-child,
      .bp3-dark .bp3-callout.bp3-intent-primary .bp3-heading{
        color:#48aff0; }
  .bp3-callout.bp3-intent-success{
    background-color:rgba(15, 153, 96, 0.15); }
    .bp3-callout.bp3-intent-success[class*="bp3-icon-"]::before,
    .bp3-callout.bp3-intent-success > .bp3-icon:first-child,
    .bp3-callout.bp3-intent-success .bp3-heading{
      color:#0d8050; }
    .bp3-dark .bp3-callout.bp3-intent-success{
      background-color:rgba(15, 153, 96, 0.25); }
      .bp3-dark .bp3-callout.bp3-intent-success[class*="bp3-icon-"]::before,
      .bp3-dark .bp3-callout.bp3-intent-success > .bp3-icon:first-child,
      .bp3-dark .bp3-callout.bp3-intent-success .bp3-heading{
        color:#3dcc91; }
  .bp3-callout.bp3-intent-warning{
    background-color:rgba(217, 130, 43, 0.15); }
    .bp3-callout.bp3-intent-warning[class*="bp3-icon-"]::before,
    .bp3-callout.bp3-intent-warning > .bp3-icon:first-child,
    .bp3-callout.bp3-intent-warning .bp3-heading{
      color:#bf7326; }
    .bp3-dark .bp3-callout.bp3-intent-warning{
      background-color:rgba(217, 130, 43, 0.25); }
      .bp3-dark .bp3-callout.bp3-intent-warning[class*="bp3-icon-"]::before,
      .bp3-dark .bp3-callout.bp3-intent-warning > .bp3-icon:first-child,
      .bp3-dark .bp3-callout.bp3-intent-warning .bp3-heading{
        color:#ffb366; }
  .bp3-callout.bp3-intent-danger{
    background-color:rgba(219, 55, 55, 0.15); }
    .bp3-callout.bp3-intent-danger[class*="bp3-icon-"]::before,
    .bp3-callout.bp3-intent-danger > .bp3-icon:first-child,
    .bp3-callout.bp3-intent-danger .bp3-heading{
      color:#c23030; }
    .bp3-dark .bp3-callout.bp3-intent-danger{
      background-color:rgba(219, 55, 55, 0.25); }
      .bp3-dark .bp3-callout.bp3-intent-danger[class*="bp3-icon-"]::before,
      .bp3-dark .bp3-callout.bp3-intent-danger > .bp3-icon:first-child,
      .bp3-dark .bp3-callout.bp3-intent-danger .bp3-heading{
        color:#ff7373; }
  .bp3-running-text .bp3-callout{
    margin:20px 0; }
.bp3-card{
  background-color:#ffffff;
  border-radius:3px;
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.15), 0 0 0 rgba(16, 22, 26, 0), 0 0 0 rgba(16, 22, 26, 0);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.15), 0 0 0 rgba(16, 22, 26, 0), 0 0 0 rgba(16, 22, 26, 0);
  padding:20px;
  -webkit-transition:-webkit-transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9), -webkit-box-shadow 200ms cubic-bezier(0.4, 1, 0.75, 0.9);
  transition:-webkit-transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9), -webkit-box-shadow 200ms cubic-bezier(0.4, 1, 0.75, 0.9);
  transition:transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9), box-shadow 200ms cubic-bezier(0.4, 1, 0.75, 0.9);
  transition:transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9), box-shadow 200ms cubic-bezier(0.4, 1, 0.75, 0.9), -webkit-transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9), -webkit-box-shadow 200ms cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-card.bp3-dark,
  .bp3-dark .bp3-card{
    background-color:#30404d;
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4), 0 0 0 rgba(16, 22, 26, 0), 0 0 0 rgba(16, 22, 26, 0);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4), 0 0 0 rgba(16, 22, 26, 0), 0 0 0 rgba(16, 22, 26, 0); }

.bp3-elevation-0{
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.15), 0 0 0 rgba(16, 22, 26, 0), 0 0 0 rgba(16, 22, 26, 0);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.15), 0 0 0 rgba(16, 22, 26, 0), 0 0 0 rgba(16, 22, 26, 0); }
  .bp3-elevation-0.bp3-dark,
  .bp3-dark .bp3-elevation-0{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4), 0 0 0 rgba(16, 22, 26, 0), 0 0 0 rgba(16, 22, 26, 0);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4), 0 0 0 rgba(16, 22, 26, 0), 0 0 0 rgba(16, 22, 26, 0); }

.bp3-elevation-1{
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.2);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.2); }
  .bp3-elevation-1.bp3-dark,
  .bp3-dark .bp3-elevation-1{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.4); }

.bp3-elevation-2{
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 1px 1px rgba(16, 22, 26, 0.2), 0 2px 6px rgba(16, 22, 26, 0.2);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 1px 1px rgba(16, 22, 26, 0.2), 0 2px 6px rgba(16, 22, 26, 0.2); }
  .bp3-elevation-2.bp3-dark,
  .bp3-dark .bp3-elevation-2{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 1px 1px rgba(16, 22, 26, 0.4), 0 2px 6px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 1px 1px rgba(16, 22, 26, 0.4), 0 2px 6px rgba(16, 22, 26, 0.4); }

.bp3-elevation-3{
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 2px 4px rgba(16, 22, 26, 0.2), 0 8px 24px rgba(16, 22, 26, 0.2);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 2px 4px rgba(16, 22, 26, 0.2), 0 8px 24px rgba(16, 22, 26, 0.2); }
  .bp3-elevation-3.bp3-dark,
  .bp3-dark .bp3-elevation-3{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 2px 4px rgba(16, 22, 26, 0.4), 0 8px 24px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 2px 4px rgba(16, 22, 26, 0.4), 0 8px 24px rgba(16, 22, 26, 0.4); }

.bp3-elevation-4{
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 4px 8px rgba(16, 22, 26, 0.2), 0 18px 46px 6px rgba(16, 22, 26, 0.2);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 4px 8px rgba(16, 22, 26, 0.2), 0 18px 46px 6px rgba(16, 22, 26, 0.2); }
  .bp3-elevation-4.bp3-dark,
  .bp3-dark .bp3-elevation-4{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 4px 8px rgba(16, 22, 26, 0.4), 0 18px 46px 6px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 4px 8px rgba(16, 22, 26, 0.4), 0 18px 46px 6px rgba(16, 22, 26, 0.4); }

.bp3-card.bp3-interactive:hover{
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 2px 4px rgba(16, 22, 26, 0.2), 0 8px 24px rgba(16, 22, 26, 0.2);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 2px 4px rgba(16, 22, 26, 0.2), 0 8px 24px rgba(16, 22, 26, 0.2);
  cursor:pointer; }
  .bp3-card.bp3-interactive:hover.bp3-dark,
  .bp3-dark .bp3-card.bp3-interactive:hover{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 2px 4px rgba(16, 22, 26, 0.4), 0 8px 24px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 2px 4px rgba(16, 22, 26, 0.4), 0 8px 24px rgba(16, 22, 26, 0.4); }

.bp3-card.bp3-interactive:active{
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.2);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.2);
  opacity:0.9;
  -webkit-transition-duration:0;
          transition-duration:0; }
  .bp3-card.bp3-interactive:active.bp3-dark,
  .bp3-dark .bp3-card.bp3-interactive:active{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.4); }

.bp3-collapse{
  height:0;
  overflow-y:hidden;
  -webkit-transition:height 200ms cubic-bezier(0.4, 1, 0.75, 0.9);
  transition:height 200ms cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-collapse .bp3-collapse-body{
    -webkit-transition:-webkit-transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9);
    transition:-webkit-transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9);
    transition:transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9);
    transition:transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9), -webkit-transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9); }
    .bp3-collapse .bp3-collapse-body[aria-hidden="true"]{
      display:none; }

.bp3-context-menu .bp3-popover-target{
  display:block; }

.bp3-context-menu-popover-target{
  position:fixed; }

.bp3-divider{
  border-bottom:1px solid rgba(16, 22, 26, 0.15);
  border-right:1px solid rgba(16, 22, 26, 0.15);
  margin:5px; }
  .bp3-dark .bp3-divider{
    border-color:rgba(16, 22, 26, 0.4); }
.bp3-dialog-container{
  opacity:1;
  -webkit-transform:scale(1);
          transform:scale(1);
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-pack:center;
      -ms-flex-pack:center;
          justify-content:center;
  min-height:100%;
  pointer-events:none;
  -webkit-user-select:none;
     -moz-user-select:none;
      -ms-user-select:none;
          user-select:none;
  width:100%; }
  .bp3-dialog-container.bp3-overlay-enter > .bp3-dialog, .bp3-dialog-container.bp3-overlay-appear > .bp3-dialog{
    opacity:0;
    -webkit-transform:scale(0.5);
            transform:scale(0.5); }
  .bp3-dialog-container.bp3-overlay-enter-active > .bp3-dialog, .bp3-dialog-container.bp3-overlay-appear-active > .bp3-dialog{
    opacity:1;
    -webkit-transform:scale(1);
            transform:scale(1);
    -webkit-transition-delay:0;
            transition-delay:0;
    -webkit-transition-duration:300ms;
            transition-duration:300ms;
    -webkit-transition-property:opacity, -webkit-transform;
    transition-property:opacity, -webkit-transform;
    transition-property:opacity, transform;
    transition-property:opacity, transform, -webkit-transform;
    -webkit-transition-timing-function:cubic-bezier(0.54, 1.12, 0.38, 1.11);
            transition-timing-function:cubic-bezier(0.54, 1.12, 0.38, 1.11); }
  .bp3-dialog-container.bp3-overlay-exit > .bp3-dialog{
    opacity:1;
    -webkit-transform:scale(1);
            transform:scale(1); }
  .bp3-dialog-container.bp3-overlay-exit-active > .bp3-dialog{
    opacity:0;
    -webkit-transform:scale(0.5);
            transform:scale(0.5);
    -webkit-transition-delay:0;
            transition-delay:0;
    -webkit-transition-duration:300ms;
            transition-duration:300ms;
    -webkit-transition-property:opacity, -webkit-transform;
    transition-property:opacity, -webkit-transform;
    transition-property:opacity, transform;
    transition-property:opacity, transform, -webkit-transform;
    -webkit-transition-timing-function:cubic-bezier(0.54, 1.12, 0.38, 1.11);
            transition-timing-function:cubic-bezier(0.54, 1.12, 0.38, 1.11); }

.bp3-dialog{
  background:#ebf1f5;
  border-radius:6px;
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 4px 8px rgba(16, 22, 26, 0.2), 0 18px 46px 6px rgba(16, 22, 26, 0.2);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 4px 8px rgba(16, 22, 26, 0.2), 0 18px 46px 6px rgba(16, 22, 26, 0.2);
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-orient:vertical;
  -webkit-box-direction:normal;
      -ms-flex-direction:column;
          flex-direction:column;
  margin:30px 0;
  padding-bottom:20px;
  pointer-events:all;
  -webkit-user-select:text;
     -moz-user-select:text;
      -ms-user-select:text;
          user-select:text;
  width:500px; }
  .bp3-dialog:focus{
    outline:0; }
  .bp3-dialog.bp3-dark,
  .bp3-dark .bp3-dialog{
    background:#293742;
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 4px 8px rgba(16, 22, 26, 0.4), 0 18px 46px 6px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 4px 8px rgba(16, 22, 26, 0.4), 0 18px 46px 6px rgba(16, 22, 26, 0.4);
    color:#f5f8fa; }

.bp3-dialog-header{
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  background:#ffffff;
  border-radius:6px 6px 0 0;
  -webkit-box-shadow:0 1px 0 rgba(16, 22, 26, 0.15);
          box-shadow:0 1px 0 rgba(16, 22, 26, 0.15);
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-flex:0;
      -ms-flex:0 0 auto;
          flex:0 0 auto;
  min-height:40px;
  padding-left:20px;
  padding-right:5px;
  z-index:30; }
  .bp3-dialog-header .bp3-icon-large,
  .bp3-dialog-header .bp3-icon{
    color:#5c7080;
    -webkit-box-flex:0;
        -ms-flex:0 0 auto;
            flex:0 0 auto;
    margin-right:10px; }
  .bp3-dialog-header .bp3-heading{
    overflow:hidden;
    text-overflow:ellipsis;
    white-space:nowrap;
    word-wrap:normal;
    -webkit-box-flex:1;
        -ms-flex:1 1 auto;
            flex:1 1 auto;
    line-height:inherit;
    margin:0; }
    .bp3-dialog-header .bp3-heading:last-child{
      margin-right:20px; }
  .bp3-dark .bp3-dialog-header{
    background:#30404d;
    -webkit-box-shadow:0 1px 0 rgba(16, 22, 26, 0.4);
            box-shadow:0 1px 0 rgba(16, 22, 26, 0.4); }
    .bp3-dark .bp3-dialog-header .bp3-icon-large,
    .bp3-dark .bp3-dialog-header .bp3-icon{
      color:#a7b6c2; }

.bp3-dialog-body{
  -webkit-box-flex:1;
      -ms-flex:1 1 auto;
          flex:1 1 auto;
  line-height:18px;
  margin:20px; }

.bp3-dialog-footer{
  -webkit-box-flex:0;
      -ms-flex:0 0 auto;
          flex:0 0 auto;
  margin:0 20px; }

.bp3-dialog-footer-actions{
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-pack:end;
      -ms-flex-pack:end;
          justify-content:flex-end; }
  .bp3-dialog-footer-actions .bp3-button{
    margin-left:10px; }
.bp3-multistep-dialog-panels{
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex; }

.bp3-multistep-dialog-left-panel{
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-flex:1;
      -ms-flex:1;
          flex:1;
  -webkit-box-orient:vertical;
  -webkit-box-direction:normal;
      -ms-flex-direction:column;
          flex-direction:column; }
  .bp3-dark .bp3-multistep-dialog-left-panel{
    background:#202b33; }

.bp3-multistep-dialog-right-panel{
  background-color:#f5f8fa;
  border-left:1px solid rgba(16, 22, 26, 0.15);
  border-radius:0 0 6px 0;
  -webkit-box-flex:3;
      -ms-flex:3;
          flex:3;
  min-width:0; }
  .bp3-dark .bp3-multistep-dialog-right-panel{
    background-color:#293742;
    border-left:1px solid rgba(16, 22, 26, 0.4); }

.bp3-multistep-dialog-footer{
  background-color:#ffffff;
  border-radius:0 0 6px 0;
  border-top:1px solid rgba(16, 22, 26, 0.15);
  padding:10px; }
  .bp3-dark .bp3-multistep-dialog-footer{
    background:#30404d;
    border-top:1px solid rgba(16, 22, 26, 0.4); }

.bp3-dialog-step-container{
  background-color:#f5f8fa;
  border-bottom:1px solid rgba(16, 22, 26, 0.15); }
  .bp3-dark .bp3-dialog-step-container{
    background:#293742;
    border-bottom:1px solid rgba(16, 22, 26, 0.4); }
  .bp3-dialog-step-container.bp3-dialog-step-viewed{
    background-color:#ffffff; }
    .bp3-dark .bp3-dialog-step-container.bp3-dialog-step-viewed{
      background:#30404d; }

.bp3-dialog-step{
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  background-color:#f5f8fa;
  border-radius:6px;
  cursor:not-allowed;
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  margin:4px;
  padding:6px 14px; }
  .bp3-dark .bp3-dialog-step{
    background:#293742; }
  .bp3-dialog-step-viewed .bp3-dialog-step{
    background-color:#ffffff;
    cursor:pointer; }
    .bp3-dark .bp3-dialog-step-viewed .bp3-dialog-step{
      background:#30404d; }
  .bp3-dialog-step:hover{
    background-color:#f5f8fa; }
    .bp3-dark .bp3-dialog-step:hover{
      background:#293742; }

.bp3-dialog-step-icon{
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  background-color:rgba(92, 112, 128, 0.6);
  border-radius:50%;
  color:#ffffff;
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  height:25px;
  -webkit-box-pack:center;
      -ms-flex-pack:center;
          justify-content:center;
  width:25px; }
  .bp3-dark .bp3-dialog-step-icon{
    background-color:rgba(167, 182, 194, 0.6); }
  .bp3-active.bp3-dialog-step-viewed .bp3-dialog-step-icon{
    background-color:#2b95d6; }
  .bp3-dialog-step-viewed .bp3-dialog-step-icon{
    background-color:#8a9ba8; }

.bp3-dialog-step-title{
  color:rgba(92, 112, 128, 0.6);
  -webkit-box-flex:1;
      -ms-flex:1;
          flex:1;
  padding-left:10px; }
  .bp3-dark .bp3-dialog-step-title{
    color:rgba(167, 182, 194, 0.6); }
  .bp3-active.bp3-dialog-step-viewed .bp3-dialog-step-title{
    color:#2b95d6; }
  .bp3-dialog-step-viewed:not(.bp3-active) .bp3-dialog-step-title{
    color:#182026; }
    .bp3-dark .bp3-dialog-step-viewed:not(.bp3-active) .bp3-dialog-step-title{
      color:#f5f8fa; }
.bp3-drawer{
  background:#ffffff;
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 4px 8px rgba(16, 22, 26, 0.2), 0 18px 46px 6px rgba(16, 22, 26, 0.2);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 4px 8px rgba(16, 22, 26, 0.2), 0 18px 46px 6px rgba(16, 22, 26, 0.2);
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-orient:vertical;
  -webkit-box-direction:normal;
      -ms-flex-direction:column;
          flex-direction:column;
  margin:0;
  padding:0; }
  .bp3-drawer:focus{
    outline:0; }
  .bp3-drawer.bp3-position-top{
    height:50%;
    left:0;
    right:0;
    top:0; }
    .bp3-drawer.bp3-position-top.bp3-overlay-enter, .bp3-drawer.bp3-position-top.bp3-overlay-appear{
      -webkit-transform:translateY(-100%);
              transform:translateY(-100%); }
    .bp3-drawer.bp3-position-top.bp3-overlay-enter-active, .bp3-drawer.bp3-position-top.bp3-overlay-appear-active{
      -webkit-transform:translateY(0);
              transform:translateY(0);
      -webkit-transition-delay:0;
              transition-delay:0;
      -webkit-transition-duration:200ms;
              transition-duration:200ms;
      -webkit-transition-property:-webkit-transform;
      transition-property:-webkit-transform;
      transition-property:transform;
      transition-property:transform, -webkit-transform;
      -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
              transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
    .bp3-drawer.bp3-position-top.bp3-overlay-exit{
      -webkit-transform:translateY(0);
              transform:translateY(0); }
    .bp3-drawer.bp3-position-top.bp3-overlay-exit-active{
      -webkit-transform:translateY(-100%);
              transform:translateY(-100%);
      -webkit-transition-delay:0;
              transition-delay:0;
      -webkit-transition-duration:100ms;
              transition-duration:100ms;
      -webkit-transition-property:-webkit-transform;
      transition-property:-webkit-transform;
      transition-property:transform;
      transition-property:transform, -webkit-transform;
      -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
              transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-drawer.bp3-position-bottom{
    bottom:0;
    height:50%;
    left:0;
    right:0; }
    .bp3-drawer.bp3-position-bottom.bp3-overlay-enter, .bp3-drawer.bp3-position-bottom.bp3-overlay-appear{
      -webkit-transform:translateY(100%);
              transform:translateY(100%); }
    .bp3-drawer.bp3-position-bottom.bp3-overlay-enter-active, .bp3-drawer.bp3-position-bottom.bp3-overlay-appear-active{
      -webkit-transform:translateY(0);
              transform:translateY(0);
      -webkit-transition-delay:0;
              transition-delay:0;
      -webkit-transition-duration:200ms;
              transition-duration:200ms;
      -webkit-transition-property:-webkit-transform;
      transition-property:-webkit-transform;
      transition-property:transform;
      transition-property:transform, -webkit-transform;
      -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
              transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
    .bp3-drawer.bp3-position-bottom.bp3-overlay-exit{
      -webkit-transform:translateY(0);
              transform:translateY(0); }
    .bp3-drawer.bp3-position-bottom.bp3-overlay-exit-active{
      -webkit-transform:translateY(100%);
              transform:translateY(100%);
      -webkit-transition-delay:0;
              transition-delay:0;
      -webkit-transition-duration:100ms;
              transition-duration:100ms;
      -webkit-transition-property:-webkit-transform;
      transition-property:-webkit-transform;
      transition-property:transform;
      transition-property:transform, -webkit-transform;
      -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
              transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-drawer.bp3-position-left{
    bottom:0;
    left:0;
    top:0;
    width:50%; }
    .bp3-drawer.bp3-position-left.bp3-overlay-enter, .bp3-drawer.bp3-position-left.bp3-overlay-appear{
      -webkit-transform:translateX(-100%);
              transform:translateX(-100%); }
    .bp3-drawer.bp3-position-left.bp3-overlay-enter-active, .bp3-drawer.bp3-position-left.bp3-overlay-appear-active{
      -webkit-transform:translateX(0);
              transform:translateX(0);
      -webkit-transition-delay:0;
              transition-delay:0;
      -webkit-transition-duration:200ms;
              transition-duration:200ms;
      -webkit-transition-property:-webkit-transform;
      transition-property:-webkit-transform;
      transition-property:transform;
      transition-property:transform, -webkit-transform;
      -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
              transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
    .bp3-drawer.bp3-position-left.bp3-overlay-exit{
      -webkit-transform:translateX(0);
              transform:translateX(0); }
    .bp3-drawer.bp3-position-left.bp3-overlay-exit-active{
      -webkit-transform:translateX(-100%);
              transform:translateX(-100%);
      -webkit-transition-delay:0;
              transition-delay:0;
      -webkit-transition-duration:100ms;
              transition-duration:100ms;
      -webkit-transition-property:-webkit-transform;
      transition-property:-webkit-transform;
      transition-property:transform;
      transition-property:transform, -webkit-transform;
      -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
              transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-drawer.bp3-position-right{
    bottom:0;
    right:0;
    top:0;
    width:50%; }
    .bp3-drawer.bp3-position-right.bp3-overlay-enter, .bp3-drawer.bp3-position-right.bp3-overlay-appear{
      -webkit-transform:translateX(100%);
              transform:translateX(100%); }
    .bp3-drawer.bp3-position-right.bp3-overlay-enter-active, .bp3-drawer.bp3-position-right.bp3-overlay-appear-active{
      -webkit-transform:translateX(0);
              transform:translateX(0);
      -webkit-transition-delay:0;
              transition-delay:0;
      -webkit-transition-duration:200ms;
              transition-duration:200ms;
      -webkit-transition-property:-webkit-transform;
      transition-property:-webkit-transform;
      transition-property:transform;
      transition-property:transform, -webkit-transform;
      -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
              transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
    .bp3-drawer.bp3-position-right.bp3-overlay-exit{
      -webkit-transform:translateX(0);
              transform:translateX(0); }
    .bp3-drawer.bp3-position-right.bp3-overlay-exit-active{
      -webkit-transform:translateX(100%);
              transform:translateX(100%);
      -webkit-transition-delay:0;
              transition-delay:0;
      -webkit-transition-duration:100ms;
              transition-duration:100ms;
      -webkit-transition-property:-webkit-transform;
      transition-property:-webkit-transform;
      transition-property:transform;
      transition-property:transform, -webkit-transform;
      -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
              transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-drawer:not(.bp3-position-top):not(.bp3-position-bottom):not(.bp3-position-left):not(
  .bp3-position-right):not(.bp3-vertical){
    bottom:0;
    right:0;
    top:0;
    width:50%; }
    .bp3-drawer:not(.bp3-position-top):not(.bp3-position-bottom):not(.bp3-position-left):not(
    .bp3-position-right):not(.bp3-vertical).bp3-overlay-enter, .bp3-drawer:not(.bp3-position-top):not(.bp3-position-bottom):not(.bp3-position-left):not(
    .bp3-position-right):not(.bp3-vertical).bp3-overlay-appear{
      -webkit-transform:translateX(100%);
              transform:translateX(100%); }
    .bp3-drawer:not(.bp3-position-top):not(.bp3-position-bottom):not(.bp3-position-left):not(
    .bp3-position-right):not(.bp3-vertical).bp3-overlay-enter-active, .bp3-drawer:not(.bp3-position-top):not(.bp3-position-bottom):not(.bp3-position-left):not(
    .bp3-position-right):not(.bp3-vertical).bp3-overlay-appear-active{
      -webkit-transform:translateX(0);
              transform:translateX(0);
      -webkit-transition-delay:0;
              transition-delay:0;
      -webkit-transition-duration:200ms;
              transition-duration:200ms;
      -webkit-transition-property:-webkit-transform;
      transition-property:-webkit-transform;
      transition-property:transform;
      transition-property:transform, -webkit-transform;
      -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
              transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
    .bp3-drawer:not(.bp3-position-top):not(.bp3-position-bottom):not(.bp3-position-left):not(
    .bp3-position-right):not(.bp3-vertical).bp3-overlay-exit{
      -webkit-transform:translateX(0);
              transform:translateX(0); }
    .bp3-drawer:not(.bp3-position-top):not(.bp3-position-bottom):not(.bp3-position-left):not(
    .bp3-position-right):not(.bp3-vertical).bp3-overlay-exit-active{
      -webkit-transform:translateX(100%);
              transform:translateX(100%);
      -webkit-transition-delay:0;
              transition-delay:0;
      -webkit-transition-duration:100ms;
              transition-duration:100ms;
      -webkit-transition-property:-webkit-transform;
      transition-property:-webkit-transform;
      transition-property:transform;
      transition-property:transform, -webkit-transform;
      -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
              transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-drawer:not(.bp3-position-top):not(.bp3-position-bottom):not(.bp3-position-left):not(
  .bp3-position-right).bp3-vertical{
    bottom:0;
    height:50%;
    left:0;
    right:0; }
    .bp3-drawer:not(.bp3-position-top):not(.bp3-position-bottom):not(.bp3-position-left):not(
    .bp3-position-right).bp3-vertical.bp3-overlay-enter, .bp3-drawer:not(.bp3-position-top):not(.bp3-position-bottom):not(.bp3-position-left):not(
    .bp3-position-right).bp3-vertical.bp3-overlay-appear{
      -webkit-transform:translateY(100%);
              transform:translateY(100%); }
    .bp3-drawer:not(.bp3-position-top):not(.bp3-position-bottom):not(.bp3-position-left):not(
    .bp3-position-right).bp3-vertical.bp3-overlay-enter-active, .bp3-drawer:not(.bp3-position-top):not(.bp3-position-bottom):not(.bp3-position-left):not(
    .bp3-position-right).bp3-vertical.bp3-overlay-appear-active{
      -webkit-transform:translateY(0);
              transform:translateY(0);
      -webkit-transition-delay:0;
              transition-delay:0;
      -webkit-transition-duration:200ms;
              transition-duration:200ms;
      -webkit-transition-property:-webkit-transform;
      transition-property:-webkit-transform;
      transition-property:transform;
      transition-property:transform, -webkit-transform;
      -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
              transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
    .bp3-drawer:not(.bp3-position-top):not(.bp3-position-bottom):not(.bp3-position-left):not(
    .bp3-position-right).bp3-vertical.bp3-overlay-exit{
      -webkit-transform:translateY(0);
              transform:translateY(0); }
    .bp3-drawer:not(.bp3-position-top):not(.bp3-position-bottom):not(.bp3-position-left):not(
    .bp3-position-right).bp3-vertical.bp3-overlay-exit-active{
      -webkit-transform:translateY(100%);
              transform:translateY(100%);
      -webkit-transition-delay:0;
              transition-delay:0;
      -webkit-transition-duration:100ms;
              transition-duration:100ms;
      -webkit-transition-property:-webkit-transform;
      transition-property:-webkit-transform;
      transition-property:transform;
      transition-property:transform, -webkit-transform;
      -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
              transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-drawer.bp3-dark,
  .bp3-dark .bp3-drawer{
    background:#30404d;
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 4px 8px rgba(16, 22, 26, 0.4), 0 18px 46px 6px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 4px 8px rgba(16, 22, 26, 0.4), 0 18px 46px 6px rgba(16, 22, 26, 0.4);
    color:#f5f8fa; }

.bp3-drawer-header{
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  border-radius:0;
  -webkit-box-shadow:0 1px 0 rgba(16, 22, 26, 0.15);
          box-shadow:0 1px 0 rgba(16, 22, 26, 0.15);
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-flex:0;
      -ms-flex:0 0 auto;
          flex:0 0 auto;
  min-height:40px;
  padding:5px;
  padding-left:20px;
  position:relative; }
  .bp3-drawer-header .bp3-icon-large,
  .bp3-drawer-header .bp3-icon{
    color:#5c7080;
    -webkit-box-flex:0;
        -ms-flex:0 0 auto;
            flex:0 0 auto;
    margin-right:10px; }
  .bp3-drawer-header .bp3-heading{
    overflow:hidden;
    text-overflow:ellipsis;
    white-space:nowrap;
    word-wrap:normal;
    -webkit-box-flex:1;
        -ms-flex:1 1 auto;
            flex:1 1 auto;
    line-height:inherit;
    margin:0; }
    .bp3-drawer-header .bp3-heading:last-child{
      margin-right:20px; }
  .bp3-dark .bp3-drawer-header{
    -webkit-box-shadow:0 1px 0 rgba(16, 22, 26, 0.4);
            box-shadow:0 1px 0 rgba(16, 22, 26, 0.4); }
    .bp3-dark .bp3-drawer-header .bp3-icon-large,
    .bp3-dark .bp3-drawer-header .bp3-icon{
      color:#a7b6c2; }

.bp3-drawer-body{
  -webkit-box-flex:1;
      -ms-flex:1 1 auto;
          flex:1 1 auto;
  line-height:18px;
  overflow:auto; }

.bp3-drawer-footer{
  -webkit-box-shadow:inset 0 1px 0 rgba(16, 22, 26, 0.15);
          box-shadow:inset 0 1px 0 rgba(16, 22, 26, 0.15);
  -webkit-box-flex:0;
      -ms-flex:0 0 auto;
          flex:0 0 auto;
  padding:10px 20px;
  position:relative; }
  .bp3-dark .bp3-drawer-footer{
    -webkit-box-shadow:inset 0 1px 0 rgba(16, 22, 26, 0.4);
            box-shadow:inset 0 1px 0 rgba(16, 22, 26, 0.4); }
.bp3-editable-text{
  cursor:text;
  display:inline-block;
  max-width:100%;
  position:relative;
  vertical-align:top;
  white-space:nowrap; }
  .bp3-editable-text::before{
    bottom:-3px;
    left:-3px;
    position:absolute;
    right:-3px;
    top:-3px;
    border-radius:3px;
    content:"";
    -webkit-transition:background-color 100ms cubic-bezier(0.4, 1, 0.75, 0.9), -webkit-box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9);
    transition:background-color 100ms cubic-bezier(0.4, 1, 0.75, 0.9), -webkit-box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9);
    transition:background-color 100ms cubic-bezier(0.4, 1, 0.75, 0.9), box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9);
    transition:background-color 100ms cubic-bezier(0.4, 1, 0.75, 0.9), box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9), -webkit-box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-editable-text:hover::before{
    -webkit-box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(16, 22, 26, 0.15);
            box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(16, 22, 26, 0.15); }
  .bp3-editable-text.bp3-editable-text-editing::before{
    background-color:#ffffff;
    -webkit-box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
  .bp3-editable-text.bp3-disabled::before{
    -webkit-box-shadow:none;
            box-shadow:none; }
  .bp3-editable-text.bp3-intent-primary .bp3-editable-text-input,
  .bp3-editable-text.bp3-intent-primary .bp3-editable-text-content{
    color:#137cbd; }
  .bp3-editable-text.bp3-intent-primary:hover::before{
    -webkit-box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(19, 124, 189, 0.4);
            box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(19, 124, 189, 0.4); }
  .bp3-editable-text.bp3-intent-primary.bp3-editable-text-editing::before{
    -webkit-box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
  .bp3-editable-text.bp3-intent-success .bp3-editable-text-input,
  .bp3-editable-text.bp3-intent-success .bp3-editable-text-content{
    color:#0f9960; }
  .bp3-editable-text.bp3-intent-success:hover::before{
    -webkit-box-shadow:0 0 0 0 rgba(15, 153, 96, 0), 0 0 0 0 rgba(15, 153, 96, 0), inset 0 0 0 1px rgba(15, 153, 96, 0.4);
            box-shadow:0 0 0 0 rgba(15, 153, 96, 0), 0 0 0 0 rgba(15, 153, 96, 0), inset 0 0 0 1px rgba(15, 153, 96, 0.4); }
  .bp3-editable-text.bp3-intent-success.bp3-editable-text-editing::before{
    -webkit-box-shadow:0 0 0 1px #0f9960, 0 0 0 3px rgba(15, 153, 96, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 1px #0f9960, 0 0 0 3px rgba(15, 153, 96, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
  .bp3-editable-text.bp3-intent-warning .bp3-editable-text-input,
  .bp3-editable-text.bp3-intent-warning .bp3-editable-text-content{
    color:#d9822b; }
  .bp3-editable-text.bp3-intent-warning:hover::before{
    -webkit-box-shadow:0 0 0 0 rgba(217, 130, 43, 0), 0 0 0 0 rgba(217, 130, 43, 0), inset 0 0 0 1px rgba(217, 130, 43, 0.4);
            box-shadow:0 0 0 0 rgba(217, 130, 43, 0), 0 0 0 0 rgba(217, 130, 43, 0), inset 0 0 0 1px rgba(217, 130, 43, 0.4); }
  .bp3-editable-text.bp3-intent-warning.bp3-editable-text-editing::before{
    -webkit-box-shadow:0 0 0 1px #d9822b, 0 0 0 3px rgba(217, 130, 43, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 1px #d9822b, 0 0 0 3px rgba(217, 130, 43, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
  .bp3-editable-text.bp3-intent-danger .bp3-editable-text-input,
  .bp3-editable-text.bp3-intent-danger .bp3-editable-text-content{
    color:#db3737; }
  .bp3-editable-text.bp3-intent-danger:hover::before{
    -webkit-box-shadow:0 0 0 0 rgba(219, 55, 55, 0), 0 0 0 0 rgba(219, 55, 55, 0), inset 0 0 0 1px rgba(219, 55, 55, 0.4);
            box-shadow:0 0 0 0 rgba(219, 55, 55, 0), 0 0 0 0 rgba(219, 55, 55, 0), inset 0 0 0 1px rgba(219, 55, 55, 0.4); }
  .bp3-editable-text.bp3-intent-danger.bp3-editable-text-editing::before{
    -webkit-box-shadow:0 0 0 1px #db3737, 0 0 0 3px rgba(219, 55, 55, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 1px #db3737, 0 0 0 3px rgba(219, 55, 55, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
  .bp3-dark .bp3-editable-text:hover::before{
    -webkit-box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(255, 255, 255, 0.15);
            box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(255, 255, 255, 0.15); }
  .bp3-dark .bp3-editable-text.bp3-editable-text-editing::before{
    background-color:rgba(16, 22, 26, 0.3);
    -webkit-box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
  .bp3-dark .bp3-editable-text.bp3-disabled::before{
    -webkit-box-shadow:none;
            box-shadow:none; }
  .bp3-dark .bp3-editable-text.bp3-intent-primary .bp3-editable-text-content{
    color:#48aff0; }
  .bp3-dark .bp3-editable-text.bp3-intent-primary:hover::before{
    -webkit-box-shadow:0 0 0 0 rgba(72, 175, 240, 0), 0 0 0 0 rgba(72, 175, 240, 0), inset 0 0 0 1px rgba(72, 175, 240, 0.4);
            box-shadow:0 0 0 0 rgba(72, 175, 240, 0), 0 0 0 0 rgba(72, 175, 240, 0), inset 0 0 0 1px rgba(72, 175, 240, 0.4); }
  .bp3-dark .bp3-editable-text.bp3-intent-primary.bp3-editable-text-editing::before{
    -webkit-box-shadow:0 0 0 1px #48aff0, 0 0 0 3px rgba(72, 175, 240, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px #48aff0, 0 0 0 3px rgba(72, 175, 240, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
  .bp3-dark .bp3-editable-text.bp3-intent-success .bp3-editable-text-content{
    color:#3dcc91; }
  .bp3-dark .bp3-editable-text.bp3-intent-success:hover::before{
    -webkit-box-shadow:0 0 0 0 rgba(61, 204, 145, 0), 0 0 0 0 rgba(61, 204, 145, 0), inset 0 0 0 1px rgba(61, 204, 145, 0.4);
            box-shadow:0 0 0 0 rgba(61, 204, 145, 0), 0 0 0 0 rgba(61, 204, 145, 0), inset 0 0 0 1px rgba(61, 204, 145, 0.4); }
  .bp3-dark .bp3-editable-text.bp3-intent-success.bp3-editable-text-editing::before{
    -webkit-box-shadow:0 0 0 1px #3dcc91, 0 0 0 3px rgba(61, 204, 145, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px #3dcc91, 0 0 0 3px rgba(61, 204, 145, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
  .bp3-dark .bp3-editable-text.bp3-intent-warning .bp3-editable-text-content{
    color:#ffb366; }
  .bp3-dark .bp3-editable-text.bp3-intent-warning:hover::before{
    -webkit-box-shadow:0 0 0 0 rgba(255, 179, 102, 0), 0 0 0 0 rgba(255, 179, 102, 0), inset 0 0 0 1px rgba(255, 179, 102, 0.4);
            box-shadow:0 0 0 0 rgba(255, 179, 102, 0), 0 0 0 0 rgba(255, 179, 102, 0), inset 0 0 0 1px rgba(255, 179, 102, 0.4); }
  .bp3-dark .bp3-editable-text.bp3-intent-warning.bp3-editable-text-editing::before{
    -webkit-box-shadow:0 0 0 1px #ffb366, 0 0 0 3px rgba(255, 179, 102, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px #ffb366, 0 0 0 3px rgba(255, 179, 102, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
  .bp3-dark .bp3-editable-text.bp3-intent-danger .bp3-editable-text-content{
    color:#ff7373; }
  .bp3-dark .bp3-editable-text.bp3-intent-danger:hover::before{
    -webkit-box-shadow:0 0 0 0 rgba(255, 115, 115, 0), 0 0 0 0 rgba(255, 115, 115, 0), inset 0 0 0 1px rgba(255, 115, 115, 0.4);
            box-shadow:0 0 0 0 rgba(255, 115, 115, 0), 0 0 0 0 rgba(255, 115, 115, 0), inset 0 0 0 1px rgba(255, 115, 115, 0.4); }
  .bp3-dark .bp3-editable-text.bp3-intent-danger.bp3-editable-text-editing::before{
    -webkit-box-shadow:0 0 0 1px #ff7373, 0 0 0 3px rgba(255, 115, 115, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px #ff7373, 0 0 0 3px rgba(255, 115, 115, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }

.bp3-editable-text-input,
.bp3-editable-text-content{
  color:inherit;
  display:inherit;
  font:inherit;
  letter-spacing:inherit;
  max-width:inherit;
  min-width:inherit;
  position:relative;
  resize:none;
  text-transform:inherit;
  vertical-align:top; }

.bp3-editable-text-input{
  background:none;
  border:none;
  -webkit-box-shadow:none;
          box-shadow:none;
  padding:0;
  white-space:pre-wrap;
  width:100%; }
  .bp3-editable-text-input::-webkit-input-placeholder{
    color:rgba(92, 112, 128, 0.6);
    opacity:1; }
  .bp3-editable-text-input::-moz-placeholder{
    color:rgba(92, 112, 128, 0.6);
    opacity:1; }
  .bp3-editable-text-input:-ms-input-placeholder{
    color:rgba(92, 112, 128, 0.6);
    opacity:1; }
  .bp3-editable-text-input::-ms-input-placeholder{
    color:rgba(92, 112, 128, 0.6);
    opacity:1; }
  .bp3-editable-text-input::placeholder{
    color:rgba(92, 112, 128, 0.6);
    opacity:1; }
  .bp3-editable-text-input:focus{
    outline:none; }
  .bp3-editable-text-input::-ms-clear{
    display:none; }

.bp3-editable-text-content{
  overflow:hidden;
  padding-right:2px;
  text-overflow:ellipsis;
  white-space:pre; }
  .bp3-editable-text-editing > .bp3-editable-text-content{
    left:0;
    position:absolute;
    visibility:hidden; }
  .bp3-editable-text-placeholder > .bp3-editable-text-content{
    color:rgba(92, 112, 128, 0.6); }
    .bp3-dark .bp3-editable-text-placeholder > .bp3-editable-text-content{
      color:rgba(167, 182, 194, 0.6); }

.bp3-editable-text.bp3-multiline{
  display:block; }
  .bp3-editable-text.bp3-multiline .bp3-editable-text-content{
    overflow:auto;
    white-space:pre-wrap;
    word-wrap:break-word; }
.bp3-divider{
  border-bottom:1px solid rgba(16, 22, 26, 0.15);
  border-right:1px solid rgba(16, 22, 26, 0.15);
  margin:5px; }
  .bp3-dark .bp3-divider{
    border-color:rgba(16, 22, 26, 0.4); }
.bp3-control-group{
  -webkit-transform:translateZ(0);
          transform:translateZ(0);
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-orient:horizontal;
  -webkit-box-direction:normal;
      -ms-flex-direction:row;
          flex-direction:row;
  -webkit-box-align:stretch;
      -ms-flex-align:stretch;
          align-items:stretch; }
  .bp3-control-group > *{
    -webkit-box-flex:0;
        -ms-flex-positive:0;
            flex-grow:0;
    -ms-flex-negative:0;
        flex-shrink:0; }
  .bp3-control-group > .bp3-fill{
    -webkit-box-flex:1;
        -ms-flex-positive:1;
            flex-grow:1;
    -ms-flex-negative:1;
        flex-shrink:1; }
  .bp3-control-group .bp3-button,
  .bp3-control-group .bp3-html-select,
  .bp3-control-group .bp3-input,
  .bp3-control-group .bp3-select{
    position:relative; }
  .bp3-control-group .bp3-input{
    border-radius:inherit;
    z-index:2; }
    .bp3-control-group .bp3-input:focus{
      border-radius:3px;
      z-index:14; }
    .bp3-control-group .bp3-input[class*="bp3-intent"]{
      z-index:13; }
      .bp3-control-group .bp3-input[class*="bp3-intent"]:focus{
        z-index:15; }
    .bp3-control-group .bp3-input[readonly], .bp3-control-group .bp3-input:disabled, .bp3-control-group .bp3-input.bp3-disabled{
      z-index:1; }
  .bp3-control-group .bp3-input-group[class*="bp3-intent"] .bp3-input{
    z-index:13; }
    .bp3-control-group .bp3-input-group[class*="bp3-intent"] .bp3-input:focus{
      z-index:15; }
  .bp3-control-group .bp3-button,
  .bp3-control-group .bp3-html-select select,
  .bp3-control-group .bp3-select select{
    -webkit-transform:translateZ(0);
            transform:translateZ(0);
    border-radius:inherit;
    z-index:4; }
    .bp3-control-group .bp3-button:focus,
    .bp3-control-group .bp3-html-select select:focus,
    .bp3-control-group .bp3-select select:focus{
      z-index:5; }
    .bp3-control-group .bp3-button:hover,
    .bp3-control-group .bp3-html-select select:hover,
    .bp3-control-group .bp3-select select:hover{
      z-index:6; }
    .bp3-control-group .bp3-button:active,
    .bp3-control-group .bp3-html-select select:active,
    .bp3-control-group .bp3-select select:active{
      z-index:7; }
    .bp3-control-group .bp3-button[readonly], .bp3-control-group .bp3-button:disabled, .bp3-control-group .bp3-button.bp3-disabled,
    .bp3-control-group .bp3-html-select select[readonly],
    .bp3-control-group .bp3-html-select select:disabled,
    .bp3-control-group .bp3-html-select select.bp3-disabled,
    .bp3-control-group .bp3-select select[readonly],
    .bp3-control-group .bp3-select select:disabled,
    .bp3-control-group .bp3-select select.bp3-disabled{
      z-index:3; }
    .bp3-control-group .bp3-button[class*="bp3-intent"],
    .bp3-control-group .bp3-html-select select[class*="bp3-intent"],
    .bp3-control-group .bp3-select select[class*="bp3-intent"]{
      z-index:9; }
      .bp3-control-group .bp3-button[class*="bp3-intent"]:focus,
      .bp3-control-group .bp3-html-select select[class*="bp3-intent"]:focus,
      .bp3-control-group .bp3-select select[class*="bp3-intent"]:focus{
        z-index:10; }
      .bp3-control-group .bp3-button[class*="bp3-intent"]:hover,
      .bp3-control-group .bp3-html-select select[class*="bp3-intent"]:hover,
      .bp3-control-group .bp3-select select[class*="bp3-intent"]:hover{
        z-index:11; }
      .bp3-control-group .bp3-button[class*="bp3-intent"]:active,
      .bp3-control-group .bp3-html-select select[class*="bp3-intent"]:active,
      .bp3-control-group .bp3-select select[class*="bp3-intent"]:active{
        z-index:12; }
      .bp3-control-group .bp3-button[class*="bp3-intent"][readonly], .bp3-control-group .bp3-button[class*="bp3-intent"]:disabled, .bp3-control-group .bp3-button[class*="bp3-intent"].bp3-disabled,
      .bp3-control-group .bp3-html-select select[class*="bp3-intent"][readonly],
      .bp3-control-group .bp3-html-select select[class*="bp3-intent"]:disabled,
      .bp3-control-group .bp3-html-select select[class*="bp3-intent"].bp3-disabled,
      .bp3-control-group .bp3-select select[class*="bp3-intent"][readonly],
      .bp3-control-group .bp3-select select[class*="bp3-intent"]:disabled,
      .bp3-control-group .bp3-select select[class*="bp3-intent"].bp3-disabled{
        z-index:8; }
  .bp3-control-group .bp3-input-group > .bp3-icon,
  .bp3-control-group .bp3-input-group > .bp3-button,
  .bp3-control-group .bp3-input-group > .bp3-input-left-container,
  .bp3-control-group .bp3-input-group > .bp3-input-action{
    z-index:16; }
  .bp3-control-group .bp3-select::after,
  .bp3-control-group .bp3-html-select::after,
  .bp3-control-group .bp3-select > .bp3-icon,
  .bp3-control-group .bp3-html-select > .bp3-icon{
    z-index:17; }
  .bp3-control-group .bp3-select:focus-within{
    z-index:5; }
  .bp3-control-group:not(.bp3-vertical) > *:not(.bp3-divider){
    margin-right:-1px; }
  .bp3-control-group:not(.bp3-vertical) > .bp3-divider:not(:first-child){
    margin-left:6px; }
  .bp3-dark .bp3-control-group:not(.bp3-vertical) > *:not(.bp3-divider){
    margin-right:0; }
  .bp3-dark .bp3-control-group:not(.bp3-vertical) > .bp3-button + .bp3-button{
    margin-left:1px; }
  .bp3-control-group .bp3-popover-wrapper,
  .bp3-control-group .bp3-popover-target{
    border-radius:inherit; }
  .bp3-control-group > :first-child{
    border-radius:3px 0 0 3px; }
  .bp3-control-group > :last-child{
    border-radius:0 3px 3px 0;
    margin-right:0; }
  .bp3-control-group > :only-child{
    border-radius:3px;
    margin-right:0; }
  .bp3-control-group .bp3-input-group .bp3-button{
    border-radius:3px; }
  .bp3-control-group .bp3-numeric-input:not(:first-child) .bp3-input-group{
    border-bottom-left-radius:0;
    border-top-left-radius:0; }
  .bp3-control-group.bp3-fill{
    width:100%; }
  .bp3-control-group > .bp3-fill{
    -webkit-box-flex:1;
        -ms-flex:1 1 auto;
            flex:1 1 auto; }
  .bp3-control-group.bp3-fill > *:not(.bp3-fixed){
    -webkit-box-flex:1;
        -ms-flex:1 1 auto;
            flex:1 1 auto; }
  .bp3-control-group.bp3-vertical{
    -webkit-box-orient:vertical;
    -webkit-box-direction:normal;
        -ms-flex-direction:column;
            flex-direction:column; }
    .bp3-control-group.bp3-vertical > *{
      margin-top:-1px; }
    .bp3-control-group.bp3-vertical > :first-child{
      border-radius:3px 3px 0 0;
      margin-top:0; }
    .bp3-control-group.bp3-vertical > :last-child{
      border-radius:0 0 3px 3px; }
.bp3-control{
  cursor:pointer;
  display:block;
  margin-bottom:10px;
  position:relative;
  text-transform:none; }
  .bp3-control input:checked ~ .bp3-control-indicator{
    background-color:#137cbd;
    background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.1)), to(rgba(255, 255, 255, 0)));
    background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0));
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
    color:#ffffff; }
  .bp3-control:hover input:checked ~ .bp3-control-indicator{
    background-color:#106ba3;
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2); }
  .bp3-control input:not(:disabled):active:checked ~ .bp3-control-indicator{
    background:#0e5a8a;
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
  .bp3-control input:disabled:checked ~ .bp3-control-indicator{
    background:rgba(19, 124, 189, 0.5);
    -webkit-box-shadow:none;
            box-shadow:none; }
  .bp3-dark .bp3-control input:checked ~ .bp3-control-indicator{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4); }
  .bp3-dark .bp3-control:hover input:checked ~ .bp3-control-indicator{
    background-color:#106ba3;
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4); }
  .bp3-dark .bp3-control input:not(:disabled):active:checked ~ .bp3-control-indicator{
    background-color:#0e5a8a;
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
  .bp3-dark .bp3-control input:disabled:checked ~ .bp3-control-indicator{
    background:rgba(14, 90, 138, 0.5);
    -webkit-box-shadow:none;
            box-shadow:none; }
  .bp3-control:not(.bp3-align-right){
    padding-left:26px; }
    .bp3-control:not(.bp3-align-right) .bp3-control-indicator{
      margin-left:-26px; }
  .bp3-control.bp3-align-right{
    padding-right:26px; }
    .bp3-control.bp3-align-right .bp3-control-indicator{
      margin-right:-26px; }
  .bp3-control.bp3-disabled{
    color:rgba(92, 112, 128, 0.6);
    cursor:not-allowed; }
  .bp3-control.bp3-inline{
    display:inline-block;
    margin-right:20px; }
  .bp3-control input{
    left:0;
    opacity:0;
    position:absolute;
    top:0;
    z-index:-1; }
  .bp3-control .bp3-control-indicator{
    background-clip:padding-box;
    background-color:#f5f8fa;
    background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.8)), to(rgba(255, 255, 255, 0)));
    background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.8), rgba(255, 255, 255, 0));
    border:none;
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
    cursor:pointer;
    display:inline-block;
    font-size:16px;
    height:1em;
    margin-right:10px;
    margin-top:-3px;
    position:relative;
    -webkit-user-select:none;
       -moz-user-select:none;
        -ms-user-select:none;
            user-select:none;
    vertical-align:middle;
    width:1em; }
    .bp3-control .bp3-control-indicator::before{
      content:"";
      display:block;
      height:1em;
      width:1em; }
  .bp3-control:hover .bp3-control-indicator{
    background-color:#ebf1f5; }
  .bp3-control input:not(:disabled):active ~ .bp3-control-indicator{
    background:#d8e1e8;
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 2px rgba(16, 22, 26, 0.2);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
  .bp3-control input:disabled ~ .bp3-control-indicator{
    background:rgba(206, 217, 224, 0.5);
    -webkit-box-shadow:none;
            box-shadow:none;
    cursor:not-allowed; }
  .bp3-control input:focus ~ .bp3-control-indicator{
    outline:rgba(19, 124, 189, 0.6) auto 2px;
    outline-offset:2px;
    -moz-outline-radius:6px; }
  .bp3-control.bp3-align-right .bp3-control-indicator{
    float:right;
    margin-left:10px;
    margin-top:1px; }
  .bp3-control.bp3-large{
    font-size:16px; }
    .bp3-control.bp3-large:not(.bp3-align-right){
      padding-left:30px; }
      .bp3-control.bp3-large:not(.bp3-align-right) .bp3-control-indicator{
        margin-left:-30px; }
    .bp3-control.bp3-large.bp3-align-right{
      padding-right:30px; }
      .bp3-control.bp3-large.bp3-align-right .bp3-control-indicator{
        margin-right:-30px; }
    .bp3-control.bp3-large .bp3-control-indicator{
      font-size:20px; }
    .bp3-control.bp3-large.bp3-align-right .bp3-control-indicator{
      margin-top:0; }
  .bp3-control.bp3-checkbox input:indeterminate ~ .bp3-control-indicator{
    background-color:#137cbd;
    background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.1)), to(rgba(255, 255, 255, 0)));
    background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0));
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
    color:#ffffff; }
  .bp3-control.bp3-checkbox:hover input:indeterminate ~ .bp3-control-indicator{
    background-color:#106ba3;
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2); }
  .bp3-control.bp3-checkbox input:not(:disabled):active:indeterminate ~ .bp3-control-indicator{
    background:#0e5a8a;
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
  .bp3-control.bp3-checkbox input:disabled:indeterminate ~ .bp3-control-indicator{
    background:rgba(19, 124, 189, 0.5);
    -webkit-box-shadow:none;
            box-shadow:none; }
  .bp3-dark .bp3-control.bp3-checkbox input:indeterminate ~ .bp3-control-indicator{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4); }
  .bp3-dark .bp3-control.bp3-checkbox:hover input:indeterminate ~ .bp3-control-indicator{
    background-color:#106ba3;
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4); }
  .bp3-dark .bp3-control.bp3-checkbox input:not(:disabled):active:indeterminate ~ .bp3-control-indicator{
    background-color:#0e5a8a;
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
  .bp3-dark .bp3-control.bp3-checkbox input:disabled:indeterminate ~ .bp3-control-indicator{
    background:rgba(14, 90, 138, 0.5);
    -webkit-box-shadow:none;
            box-shadow:none; }
  .bp3-control.bp3-checkbox .bp3-control-indicator{
    border-radius:3px; }
  .bp3-control.bp3-checkbox input:checked ~ .bp3-control-indicator::before{
    background-image:url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'%3e%3cpath fill-rule='evenodd' clip-rule='evenodd' d='M12 5c-.28 0-.53.11-.71.29L7 9.59l-2.29-2.3a1.003 1.003 0 00-1.42 1.42l3 3c.18.18.43.29.71.29s.53-.11.71-.29l5-5A1.003 1.003 0 0012 5z' fill='white'/%3e%3c/svg%3e"); }
  .bp3-control.bp3-checkbox input:indeterminate ~ .bp3-control-indicator::before{
    background-image:url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'%3e%3cpath fill-rule='evenodd' clip-rule='evenodd' d='M11 7H5c-.55 0-1 .45-1 1s.45 1 1 1h6c.55 0 1-.45 1-1s-.45-1-1-1z' fill='white'/%3e%3c/svg%3e"); }
  .bp3-control.bp3-radio .bp3-control-indicator{
    border-radius:50%; }
  .bp3-control.bp3-radio input:checked ~ .bp3-control-indicator::before{
    background-image:radial-gradient(#ffffff, #ffffff 28%, transparent 32%); }
  .bp3-control.bp3-radio input:checked:disabled ~ .bp3-control-indicator::before{
    opacity:0.5; }
  .bp3-control.bp3-radio input:focus ~ .bp3-control-indicator{
    -moz-outline-radius:16px; }
  .bp3-control.bp3-switch input ~ .bp3-control-indicator{
    background:rgba(167, 182, 194, 0.5); }
  .bp3-control.bp3-switch:hover input ~ .bp3-control-indicator{
    background:rgba(115, 134, 148, 0.5); }
  .bp3-control.bp3-switch input:not(:disabled):active ~ .bp3-control-indicator{
    background:rgba(92, 112, 128, 0.5); }
  .bp3-control.bp3-switch input:disabled ~ .bp3-control-indicator{
    background:rgba(206, 217, 224, 0.5); }
    .bp3-control.bp3-switch input:disabled ~ .bp3-control-indicator::before{
      background:rgba(255, 255, 255, 0.8); }
  .bp3-control.bp3-switch input:checked ~ .bp3-control-indicator{
    background:#137cbd; }
  .bp3-control.bp3-switch:hover input:checked ~ .bp3-control-indicator{
    background:#106ba3; }
  .bp3-control.bp3-switch input:checked:not(:disabled):active ~ .bp3-control-indicator{
    background:#0e5a8a; }
  .bp3-control.bp3-switch input:checked:disabled ~ .bp3-control-indicator{
    background:rgba(19, 124, 189, 0.5); }
    .bp3-control.bp3-switch input:checked:disabled ~ .bp3-control-indicator::before{
      background:rgba(255, 255, 255, 0.8); }
  .bp3-control.bp3-switch:not(.bp3-align-right){
    padding-left:38px; }
    .bp3-control.bp3-switch:not(.bp3-align-right) .bp3-control-indicator{
      margin-left:-38px; }
  .bp3-control.bp3-switch.bp3-align-right{
    padding-right:38px; }
    .bp3-control.bp3-switch.bp3-align-right .bp3-control-indicator{
      margin-right:-38px; }
  .bp3-control.bp3-switch .bp3-control-indicator{
    border:none;
    border-radius:1.75em;
    -webkit-box-shadow:none !important;
            box-shadow:none !important;
    min-width:1.75em;
    -webkit-transition:background-color 100ms cubic-bezier(0.4, 1, 0.75, 0.9);
    transition:background-color 100ms cubic-bezier(0.4, 1, 0.75, 0.9);
    width:auto; }
    .bp3-control.bp3-switch .bp3-control-indicator::before{
      background:#ffffff;
      border-radius:50%;
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 1px 1px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 1px 1px rgba(16, 22, 26, 0.2);
      height:calc(1em - 4px);
      left:0;
      margin:2px;
      position:absolute;
      -webkit-transition:left 100ms cubic-bezier(0.4, 1, 0.75, 0.9);
      transition:left 100ms cubic-bezier(0.4, 1, 0.75, 0.9);
      width:calc(1em - 4px); }
  .bp3-control.bp3-switch input:checked ~ .bp3-control-indicator::before{
    left:calc(100% - 1em); }
  .bp3-control.bp3-switch.bp3-large:not(.bp3-align-right){
    padding-left:45px; }
    .bp3-control.bp3-switch.bp3-large:not(.bp3-align-right) .bp3-control-indicator{
      margin-left:-45px; }
  .bp3-control.bp3-switch.bp3-large.bp3-align-right{
    padding-right:45px; }
    .bp3-control.bp3-switch.bp3-large.bp3-align-right .bp3-control-indicator{
      margin-right:-45px; }
  .bp3-dark .bp3-control.bp3-switch input ~ .bp3-control-indicator{
    background:rgba(16, 22, 26, 0.5); }
  .bp3-dark .bp3-control.bp3-switch:hover input ~ .bp3-control-indicator{
    background:rgba(16, 22, 26, 0.7); }
  .bp3-dark .bp3-control.bp3-switch input:not(:disabled):active ~ .bp3-control-indicator{
    background:rgba(16, 22, 26, 0.9); }
  .bp3-dark .bp3-control.bp3-switch input:disabled ~ .bp3-control-indicator{
    background:rgba(57, 75, 89, 0.5); }
    .bp3-dark .bp3-control.bp3-switch input:disabled ~ .bp3-control-indicator::before{
      background:rgba(16, 22, 26, 0.4); }
  .bp3-dark .bp3-control.bp3-switch input:checked ~ .bp3-control-indicator{
    background:#137cbd; }
  .bp3-dark .bp3-control.bp3-switch:hover input:checked ~ .bp3-control-indicator{
    background:#106ba3; }
  .bp3-dark .bp3-control.bp3-switch input:checked:not(:disabled):active ~ .bp3-control-indicator{
    background:#0e5a8a; }
  .bp3-dark .bp3-control.bp3-switch input:checked:disabled ~ .bp3-control-indicator{
    background:rgba(14, 90, 138, 0.5); }
    .bp3-dark .bp3-control.bp3-switch input:checked:disabled ~ .bp3-control-indicator::before{
      background:rgba(16, 22, 26, 0.4); }
  .bp3-dark .bp3-control.bp3-switch .bp3-control-indicator::before{
    background:#394b59;
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4); }
  .bp3-dark .bp3-control.bp3-switch input:checked ~ .bp3-control-indicator::before{
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4); }
  .bp3-control.bp3-switch .bp3-switch-inner-text{
    font-size:0.7em;
    text-align:center; }
  .bp3-control.bp3-switch .bp3-control-indicator-child:first-child{
    line-height:0;
    margin-left:0.5em;
    margin-right:1.2em;
    visibility:hidden; }
  .bp3-control.bp3-switch .bp3-control-indicator-child:last-child{
    line-height:1em;
    margin-left:1.2em;
    margin-right:0.5em;
    visibility:visible; }
  .bp3-control.bp3-switch input:checked ~ .bp3-control-indicator .bp3-control-indicator-child:first-child{
    line-height:1em;
    visibility:visible; }
  .bp3-control.bp3-switch input:checked ~ .bp3-control-indicator .bp3-control-indicator-child:last-child{
    line-height:0;
    visibility:hidden; }
  .bp3-dark .bp3-control{
    color:#f5f8fa; }
    .bp3-dark .bp3-control.bp3-disabled{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-control .bp3-control-indicator{
      background-color:#394b59;
      background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.05)), to(rgba(255, 255, 255, 0)));
      background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0));
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4); }
    .bp3-dark .bp3-control:hover .bp3-control-indicator{
      background-color:#30404d; }
    .bp3-dark .bp3-control input:not(:disabled):active ~ .bp3-control-indicator{
      background:#202b33;
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.6), inset 0 1px 2px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.6), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
    .bp3-dark .bp3-control input:disabled ~ .bp3-control-indicator{
      background:rgba(57, 75, 89, 0.5);
      -webkit-box-shadow:none;
              box-shadow:none;
      cursor:not-allowed; }
    .bp3-dark .bp3-control.bp3-checkbox input:disabled:checked ~ .bp3-control-indicator, .bp3-dark .bp3-control.bp3-checkbox input:disabled:indeterminate ~ .bp3-control-indicator{
      color:rgba(167, 182, 194, 0.6); }
.bp3-file-input{
  cursor:pointer;
  display:inline-block;
  height:30px;
  position:relative; }
  .bp3-file-input input{
    margin:0;
    min-width:200px;
    opacity:0; }
    .bp3-file-input input:disabled + .bp3-file-upload-input,
    .bp3-file-input input.bp3-disabled + .bp3-file-upload-input{
      background:rgba(206, 217, 224, 0.5);
      -webkit-box-shadow:none;
              box-shadow:none;
      color:rgba(92, 112, 128, 0.6);
      cursor:not-allowed;
      resize:none; }
      .bp3-file-input input:disabled + .bp3-file-upload-input::after,
      .bp3-file-input input.bp3-disabled + .bp3-file-upload-input::after{
        background-color:rgba(206, 217, 224, 0.5);
        background-image:none;
        -webkit-box-shadow:none;
                box-shadow:none;
        color:rgba(92, 112, 128, 0.6);
        cursor:not-allowed;
        outline:none; }
        .bp3-file-input input:disabled + .bp3-file-upload-input::after.bp3-active, .bp3-file-input input:disabled + .bp3-file-upload-input::after.bp3-active:hover,
        .bp3-file-input input.bp3-disabled + .bp3-file-upload-input::after.bp3-active,
        .bp3-file-input input.bp3-disabled + .bp3-file-upload-input::after.bp3-active:hover{
          background:rgba(206, 217, 224, 0.7); }
      .bp3-dark .bp3-file-input input:disabled + .bp3-file-upload-input, .bp3-dark
      .bp3-file-input input.bp3-disabled + .bp3-file-upload-input{
        background:rgba(57, 75, 89, 0.5);
        -webkit-box-shadow:none;
                box-shadow:none;
        color:rgba(167, 182, 194, 0.6); }
        .bp3-dark .bp3-file-input input:disabled + .bp3-file-upload-input::after, .bp3-dark
        .bp3-file-input input.bp3-disabled + .bp3-file-upload-input::after{
          background-color:rgba(57, 75, 89, 0.5);
          background-image:none;
          -webkit-box-shadow:none;
                  box-shadow:none;
          color:rgba(167, 182, 194, 0.6); }
          .bp3-dark .bp3-file-input input:disabled + .bp3-file-upload-input::after.bp3-active, .bp3-dark
          .bp3-file-input input.bp3-disabled + .bp3-file-upload-input::after.bp3-active{
            background:rgba(57, 75, 89, 0.7); }
  .bp3-file-input.bp3-file-input-has-selection .bp3-file-upload-input{
    color:#182026; }
  .bp3-dark .bp3-file-input.bp3-file-input-has-selection .bp3-file-upload-input{
    color:#f5f8fa; }
  .bp3-file-input.bp3-fill{
    width:100%; }
  .bp3-file-input.bp3-large,
  .bp3-large .bp3-file-input{
    height:40px; }
  .bp3-file-input .bp3-file-upload-input-custom-text::after{
    content:attr(bp3-button-text); }

.bp3-file-upload-input{
  -webkit-appearance:none;
     -moz-appearance:none;
          appearance:none;
  background:#ffffff;
  border:none;
  border-radius:3px;
  -webkit-box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2);
          box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2);
  color:#182026;
  font-size:14px;
  font-weight:400;
  height:30px;
  line-height:30px;
  outline:none;
  padding:0 10px;
  -webkit-transition:-webkit-box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9);
  transition:-webkit-box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9);
  transition:box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9);
  transition:box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9), -webkit-box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9);
  vertical-align:middle;
  overflow:hidden;
  text-overflow:ellipsis;
  white-space:nowrap;
  word-wrap:normal;
  color:rgba(92, 112, 128, 0.6);
  left:0;
  padding-right:80px;
  position:absolute;
  right:0;
  top:0;
  -webkit-user-select:none;
     -moz-user-select:none;
      -ms-user-select:none;
          user-select:none; }
  .bp3-file-upload-input::-webkit-input-placeholder{
    color:rgba(92, 112, 128, 0.6);
    opacity:1; }
  .bp3-file-upload-input::-moz-placeholder{
    color:rgba(92, 112, 128, 0.6);
    opacity:1; }
  .bp3-file-upload-input:-ms-input-placeholder{
    color:rgba(92, 112, 128, 0.6);
    opacity:1; }
  .bp3-file-upload-input::-ms-input-placeholder{
    color:rgba(92, 112, 128, 0.6);
    opacity:1; }
  .bp3-file-upload-input::placeholder{
    color:rgba(92, 112, 128, 0.6);
    opacity:1; }
  .bp3-file-upload-input:focus, .bp3-file-upload-input.bp3-active{
    -webkit-box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
  .bp3-file-upload-input[type="search"], .bp3-file-upload-input.bp3-round{
    border-radius:30px;
    -webkit-box-sizing:border-box;
            box-sizing:border-box;
    padding-left:10px; }
  .bp3-file-upload-input[readonly]{
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.15);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.15); }
  .bp3-file-upload-input:disabled, .bp3-file-upload-input.bp3-disabled{
    background:rgba(206, 217, 224, 0.5);
    -webkit-box-shadow:none;
            box-shadow:none;
    color:rgba(92, 112, 128, 0.6);
    cursor:not-allowed;
    resize:none; }
  .bp3-file-upload-input::after{
    background-color:#f5f8fa;
    background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.8)), to(rgba(255, 255, 255, 0)));
    background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.8), rgba(255, 255, 255, 0));
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
    color:#182026;
    min-height:24px;
    min-width:24px;
    overflow:hidden;
    text-overflow:ellipsis;
    white-space:nowrap;
    word-wrap:normal;
    border-radius:3px;
    content:"Browse";
    line-height:24px;
    margin:3px;
    position:absolute;
    right:0;
    text-align:center;
    top:0;
    width:70px; }
    .bp3-file-upload-input::after:hover{
      background-clip:padding-box;
      background-color:#ebf1f5;
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1); }
    .bp3-file-upload-input::after:active, .bp3-file-upload-input::after.bp3-active{
      background-color:#d8e1e8;
      background-image:none;
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 2px rgba(16, 22, 26, 0.2);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
    .bp3-file-upload-input::after:disabled, .bp3-file-upload-input::after.bp3-disabled{
      background-color:rgba(206, 217, 224, 0.5);
      background-image:none;
      -webkit-box-shadow:none;
              box-shadow:none;
      color:rgba(92, 112, 128, 0.6);
      cursor:not-allowed;
      outline:none; }
      .bp3-file-upload-input::after:disabled.bp3-active, .bp3-file-upload-input::after:disabled.bp3-active:hover, .bp3-file-upload-input::after.bp3-disabled.bp3-active, .bp3-file-upload-input::after.bp3-disabled.bp3-active:hover{
        background:rgba(206, 217, 224, 0.7); }
  .bp3-file-upload-input:hover::after{
    background-clip:padding-box;
    background-color:#ebf1f5;
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1); }
  .bp3-file-upload-input:active::after{
    background-color:#d8e1e8;
    background-image:none;
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 2px rgba(16, 22, 26, 0.2);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
  .bp3-large .bp3-file-upload-input{
    font-size:16px;
    height:40px;
    line-height:40px;
    padding-right:95px; }
    .bp3-large .bp3-file-upload-input[type="search"], .bp3-large .bp3-file-upload-input.bp3-round{
      padding:0 15px; }
    .bp3-large .bp3-file-upload-input::after{
      min-height:30px;
      min-width:30px;
      line-height:30px;
      margin:5px;
      width:85px; }
  .bp3-dark .bp3-file-upload-input{
    background:rgba(16, 22, 26, 0.3);
    -webkit-box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
    color:#f5f8fa;
    color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-file-upload-input::-webkit-input-placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-file-upload-input::-moz-placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-file-upload-input:-ms-input-placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-file-upload-input::-ms-input-placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-file-upload-input::placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-file-upload-input:focus{
      -webkit-box-shadow:0 0 0 1px #137cbd, 0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px #137cbd, 0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
    .bp3-dark .bp3-file-upload-input[readonly]{
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4); }
    .bp3-dark .bp3-file-upload-input:disabled, .bp3-dark .bp3-file-upload-input.bp3-disabled{
      background:rgba(57, 75, 89, 0.5);
      -webkit-box-shadow:none;
              box-shadow:none;
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-file-upload-input::after{
      background-color:#394b59;
      background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.05)), to(rgba(255, 255, 255, 0)));
      background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0));
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
      color:#f5f8fa; }
      .bp3-dark .bp3-file-upload-input::after:hover, .bp3-dark .bp3-file-upload-input::after:active, .bp3-dark .bp3-file-upload-input::after.bp3-active{
        color:#f5f8fa; }
      .bp3-dark .bp3-file-upload-input::after:hover{
        background-color:#30404d;
        -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
                box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4); }
      .bp3-dark .bp3-file-upload-input::after:active, .bp3-dark .bp3-file-upload-input::after.bp3-active{
        background-color:#202b33;
        background-image:none;
        -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.6), inset 0 1px 2px rgba(16, 22, 26, 0.2);
                box-shadow:0 0 0 1px rgba(16, 22, 26, 0.6), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
      .bp3-dark .bp3-file-upload-input::after:disabled, .bp3-dark .bp3-file-upload-input::after.bp3-disabled{
        background-color:rgba(57, 75, 89, 0.5);
        background-image:none;
        -webkit-box-shadow:none;
                box-shadow:none;
        color:rgba(167, 182, 194, 0.6); }
        .bp3-dark .bp3-file-upload-input::after:disabled.bp3-active, .bp3-dark .bp3-file-upload-input::after.bp3-disabled.bp3-active{
          background:rgba(57, 75, 89, 0.7); }
      .bp3-dark .bp3-file-upload-input::after .bp3-button-spinner .bp3-spinner-head{
        background:rgba(16, 22, 26, 0.5);
        stroke:#8a9ba8; }
    .bp3-dark .bp3-file-upload-input:hover::after{
      background-color:#30404d;
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4); }
    .bp3-dark .bp3-file-upload-input:active::after{
      background-color:#202b33;
      background-image:none;
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.6), inset 0 1px 2px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.6), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
.bp3-file-upload-input::after{
  -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
          box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1); }
.bp3-form-group{
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-orient:vertical;
  -webkit-box-direction:normal;
      -ms-flex-direction:column;
          flex-direction:column;
  margin:0 0 15px; }
  .bp3-form-group label.bp3-label{
    margin-bottom:5px; }
  .bp3-form-group .bp3-control{
    margin-top:7px; }
  .bp3-form-group .bp3-form-helper-text{
    color:#5c7080;
    font-size:12px;
    margin-top:5px; }
  .bp3-form-group.bp3-intent-primary .bp3-form-helper-text{
    color:#106ba3; }
  .bp3-form-group.bp3-intent-success .bp3-form-helper-text{
    color:#0d8050; }
  .bp3-form-group.bp3-intent-warning .bp3-form-helper-text{
    color:#bf7326; }
  .bp3-form-group.bp3-intent-danger .bp3-form-helper-text{
    color:#c23030; }
  .bp3-form-group.bp3-inline{
    -webkit-box-align:start;
        -ms-flex-align:start;
            align-items:flex-start;
    -webkit-box-orient:horizontal;
    -webkit-box-direction:normal;
        -ms-flex-direction:row;
            flex-direction:row; }
    .bp3-form-group.bp3-inline.bp3-large label.bp3-label{
      line-height:40px;
      margin:0 10px 0 0; }
    .bp3-form-group.bp3-inline label.bp3-label{
      line-height:30px;
      margin:0 10px 0 0; }
  .bp3-form-group.bp3-disabled .bp3-label,
  .bp3-form-group.bp3-disabled .bp3-text-muted,
  .bp3-form-group.bp3-disabled .bp3-form-helper-text{
    color:rgba(92, 112, 128, 0.6) !important; }
  .bp3-dark .bp3-form-group.bp3-intent-primary .bp3-form-helper-text{
    color:#48aff0; }
  .bp3-dark .bp3-form-group.bp3-intent-success .bp3-form-helper-text{
    color:#3dcc91; }
  .bp3-dark .bp3-form-group.bp3-intent-warning .bp3-form-helper-text{
    color:#ffb366; }
  .bp3-dark .bp3-form-group.bp3-intent-danger .bp3-form-helper-text{
    color:#ff7373; }
  .bp3-dark .bp3-form-group .bp3-form-helper-text{
    color:#a7b6c2; }
  .bp3-dark .bp3-form-group.bp3-disabled .bp3-label,
  .bp3-dark .bp3-form-group.bp3-disabled .bp3-text-muted,
  .bp3-dark .bp3-form-group.bp3-disabled .bp3-form-helper-text{
    color:rgba(167, 182, 194, 0.6) !important; }
.bp3-input-group{
  display:block;
  position:relative; }
  .bp3-input-group .bp3-input{
    position:relative;
    width:100%; }
    .bp3-input-group .bp3-input:not(:first-child){
      padding-left:30px; }
    .bp3-input-group .bp3-input:not(:last-child){
      padding-right:30px; }
  .bp3-input-group .bp3-input-action,
  .bp3-input-group > .bp3-input-left-container,
  .bp3-input-group > .bp3-button,
  .bp3-input-group > .bp3-icon{
    position:absolute;
    top:0; }
    .bp3-input-group .bp3-input-action:first-child,
    .bp3-input-group > .bp3-input-left-container:first-child,
    .bp3-input-group > .bp3-button:first-child,
    .bp3-input-group > .bp3-icon:first-child{
      left:0; }
    .bp3-input-group .bp3-input-action:last-child,
    .bp3-input-group > .bp3-input-left-container:last-child,
    .bp3-input-group > .bp3-button:last-child,
    .bp3-input-group > .bp3-icon:last-child{
      right:0; }
  .bp3-input-group .bp3-button{
    min-height:24px;
    min-width:24px;
    margin:3px;
    padding:0 7px; }
    .bp3-input-group .bp3-button:empty{
      padding:0; }
  .bp3-input-group > .bp3-input-left-container,
  .bp3-input-group > .bp3-icon{
    z-index:1; }
  .bp3-input-group > .bp3-input-left-container > .bp3-icon,
  .bp3-input-group > .bp3-icon{
    color:#5c7080; }
    .bp3-input-group > .bp3-input-left-container > .bp3-icon:empty,
    .bp3-input-group > .bp3-icon:empty{
      font-family:"Icons16", sans-serif;
      font-size:16px;
      font-style:normal;
      font-weight:400;
      line-height:1;
      -moz-osx-font-smoothing:grayscale;
      -webkit-font-smoothing:antialiased; }
  .bp3-input-group > .bp3-input-left-container > .bp3-icon,
  .bp3-input-group > .bp3-icon,
  .bp3-input-group .bp3-input-action > .bp3-spinner{
    margin:7px; }
  .bp3-input-group .bp3-tag{
    margin:5px; }
  .bp3-input-group .bp3-input:not(:focus) + .bp3-button.bp3-minimal:not(:hover):not(:focus),
  .bp3-input-group .bp3-input:not(:focus) + .bp3-input-action .bp3-button.bp3-minimal:not(:hover):not(:focus){
    color:#5c7080; }
    .bp3-dark .bp3-input-group .bp3-input:not(:focus) + .bp3-button.bp3-minimal:not(:hover):not(:focus), .bp3-dark
    .bp3-input-group .bp3-input:not(:focus) + .bp3-input-action .bp3-button.bp3-minimal:not(:hover):not(:focus){
      color:#a7b6c2; }
    .bp3-input-group .bp3-input:not(:focus) + .bp3-button.bp3-minimal:not(:hover):not(:focus) .bp3-icon, .bp3-input-group .bp3-input:not(:focus) + .bp3-button.bp3-minimal:not(:hover):not(:focus) .bp3-icon-standard, .bp3-input-group .bp3-input:not(:focus) + .bp3-button.bp3-minimal:not(:hover):not(:focus) .bp3-icon-large,
    .bp3-input-group .bp3-input:not(:focus) + .bp3-input-action .bp3-button.bp3-minimal:not(:hover):not(:focus) .bp3-icon,
    .bp3-input-group .bp3-input:not(:focus) + .bp3-input-action .bp3-button.bp3-minimal:not(:hover):not(:focus) .bp3-icon-standard,
    .bp3-input-group .bp3-input:not(:focus) + .bp3-input-action .bp3-button.bp3-minimal:not(:hover):not(:focus) .bp3-icon-large{
      color:#5c7080; }
  .bp3-input-group .bp3-input:not(:focus) + .bp3-button.bp3-minimal:disabled,
  .bp3-input-group .bp3-input:not(:focus) + .bp3-input-action .bp3-button.bp3-minimal:disabled{
    color:rgba(92, 112, 128, 0.6) !important; }
    .bp3-input-group .bp3-input:not(:focus) + .bp3-button.bp3-minimal:disabled .bp3-icon, .bp3-input-group .bp3-input:not(:focus) + .bp3-button.bp3-minimal:disabled .bp3-icon-standard, .bp3-input-group .bp3-input:not(:focus) + .bp3-button.bp3-minimal:disabled .bp3-icon-large,
    .bp3-input-group .bp3-input:not(:focus) + .bp3-input-action .bp3-button.bp3-minimal:disabled .bp3-icon,
    .bp3-input-group .bp3-input:not(:focus) + .bp3-input-action .bp3-button.bp3-minimal:disabled .bp3-icon-standard,
    .bp3-input-group .bp3-input:not(:focus) + .bp3-input-action .bp3-button.bp3-minimal:disabled .bp3-icon-large{
      color:rgba(92, 112, 128, 0.6) !important; }
  .bp3-input-group.bp3-disabled{
    cursor:not-allowed; }
    .bp3-input-group.bp3-disabled .bp3-icon{
      color:rgba(92, 112, 128, 0.6); }
  .bp3-input-group.bp3-large .bp3-button{
    min-height:30px;
    min-width:30px;
    margin:5px; }
  .bp3-input-group.bp3-large > .bp3-input-left-container > .bp3-icon,
  .bp3-input-group.bp3-large > .bp3-icon,
  .bp3-input-group.bp3-large .bp3-input-action > .bp3-spinner{
    margin:12px; }
  .bp3-input-group.bp3-large .bp3-input{
    font-size:16px;
    height:40px;
    line-height:40px; }
    .bp3-input-group.bp3-large .bp3-input[type="search"], .bp3-input-group.bp3-large .bp3-input.bp3-round{
      padding:0 15px; }
    .bp3-input-group.bp3-large .bp3-input:not(:first-child){
      padding-left:40px; }
    .bp3-input-group.bp3-large .bp3-input:not(:last-child){
      padding-right:40px; }
  .bp3-input-group.bp3-small .bp3-button{
    min-height:20px;
    min-width:20px;
    margin:2px; }
  .bp3-input-group.bp3-small .bp3-tag{
    min-height:20px;
    min-width:20px;
    margin:2px; }
  .bp3-input-group.bp3-small > .bp3-input-left-container > .bp3-icon,
  .bp3-input-group.bp3-small > .bp3-icon,
  .bp3-input-group.bp3-small .bp3-input-action > .bp3-spinner{
    margin:4px; }
  .bp3-input-group.bp3-small .bp3-input{
    font-size:12px;
    height:24px;
    line-height:24px;
    padding-left:8px;
    padding-right:8px; }
    .bp3-input-group.bp3-small .bp3-input[type="search"], .bp3-input-group.bp3-small .bp3-input.bp3-round{
      padding:0 12px; }
    .bp3-input-group.bp3-small .bp3-input:not(:first-child){
      padding-left:24px; }
    .bp3-input-group.bp3-small .bp3-input:not(:last-child){
      padding-right:24px; }
  .bp3-input-group.bp3-fill{
    -webkit-box-flex:1;
        -ms-flex:1 1 auto;
            flex:1 1 auto;
    width:100%; }
  .bp3-input-group.bp3-round .bp3-button,
  .bp3-input-group.bp3-round .bp3-input,
  .bp3-input-group.bp3-round .bp3-tag{
    border-radius:30px; }
  .bp3-dark .bp3-input-group .bp3-icon{
    color:#a7b6c2; }
  .bp3-dark .bp3-input-group.bp3-disabled .bp3-icon{
    color:rgba(167, 182, 194, 0.6); }
  .bp3-input-group.bp3-intent-primary .bp3-input{
    -webkit-box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px #137cbd, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px #137cbd, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-input-group.bp3-intent-primary .bp3-input:focus{
      -webkit-box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-input-group.bp3-intent-primary .bp3-input[readonly]{
      -webkit-box-shadow:inset 0 0 0 1px #137cbd;
              box-shadow:inset 0 0 0 1px #137cbd; }
    .bp3-input-group.bp3-intent-primary .bp3-input:disabled, .bp3-input-group.bp3-intent-primary .bp3-input.bp3-disabled{
      -webkit-box-shadow:none;
              box-shadow:none; }
  .bp3-input-group.bp3-intent-primary > .bp3-icon{
    color:#106ba3; }
    .bp3-dark .bp3-input-group.bp3-intent-primary > .bp3-icon{
      color:#48aff0; }
  .bp3-input-group.bp3-intent-success .bp3-input{
    -webkit-box-shadow:0 0 0 0 rgba(15, 153, 96, 0), 0 0 0 0 rgba(15, 153, 96, 0), inset 0 0 0 1px #0f9960, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 0 rgba(15, 153, 96, 0), 0 0 0 0 rgba(15, 153, 96, 0), inset 0 0 0 1px #0f9960, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-input-group.bp3-intent-success .bp3-input:focus{
      -webkit-box-shadow:0 0 0 1px #0f9960, 0 0 0 3px rgba(15, 153, 96, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px #0f9960, 0 0 0 3px rgba(15, 153, 96, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-input-group.bp3-intent-success .bp3-input[readonly]{
      -webkit-box-shadow:inset 0 0 0 1px #0f9960;
              box-shadow:inset 0 0 0 1px #0f9960; }
    .bp3-input-group.bp3-intent-success .bp3-input:disabled, .bp3-input-group.bp3-intent-success .bp3-input.bp3-disabled{
      -webkit-box-shadow:none;
              box-shadow:none; }
  .bp3-input-group.bp3-intent-success > .bp3-icon{
    color:#0d8050; }
    .bp3-dark .bp3-input-group.bp3-intent-success > .bp3-icon{
      color:#3dcc91; }
  .bp3-input-group.bp3-intent-warning .bp3-input{
    -webkit-box-shadow:0 0 0 0 rgba(217, 130, 43, 0), 0 0 0 0 rgba(217, 130, 43, 0), inset 0 0 0 1px #d9822b, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 0 rgba(217, 130, 43, 0), 0 0 0 0 rgba(217, 130, 43, 0), inset 0 0 0 1px #d9822b, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-input-group.bp3-intent-warning .bp3-input:focus{
      -webkit-box-shadow:0 0 0 1px #d9822b, 0 0 0 3px rgba(217, 130, 43, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px #d9822b, 0 0 0 3px rgba(217, 130, 43, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-input-group.bp3-intent-warning .bp3-input[readonly]{
      -webkit-box-shadow:inset 0 0 0 1px #d9822b;
              box-shadow:inset 0 0 0 1px #d9822b; }
    .bp3-input-group.bp3-intent-warning .bp3-input:disabled, .bp3-input-group.bp3-intent-warning .bp3-input.bp3-disabled{
      -webkit-box-shadow:none;
              box-shadow:none; }
  .bp3-input-group.bp3-intent-warning > .bp3-icon{
    color:#bf7326; }
    .bp3-dark .bp3-input-group.bp3-intent-warning > .bp3-icon{
      color:#ffb366; }
  .bp3-input-group.bp3-intent-danger .bp3-input{
    -webkit-box-shadow:0 0 0 0 rgba(219, 55, 55, 0), 0 0 0 0 rgba(219, 55, 55, 0), inset 0 0 0 1px #db3737, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 0 rgba(219, 55, 55, 0), 0 0 0 0 rgba(219, 55, 55, 0), inset 0 0 0 1px #db3737, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-input-group.bp3-intent-danger .bp3-input:focus{
      -webkit-box-shadow:0 0 0 1px #db3737, 0 0 0 3px rgba(219, 55, 55, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px #db3737, 0 0 0 3px rgba(219, 55, 55, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-input-group.bp3-intent-danger .bp3-input[readonly]{
      -webkit-box-shadow:inset 0 0 0 1px #db3737;
              box-shadow:inset 0 0 0 1px #db3737; }
    .bp3-input-group.bp3-intent-danger .bp3-input:disabled, .bp3-input-group.bp3-intent-danger .bp3-input.bp3-disabled{
      -webkit-box-shadow:none;
              box-shadow:none; }
  .bp3-input-group.bp3-intent-danger > .bp3-icon{
    color:#c23030; }
    .bp3-dark .bp3-input-group.bp3-intent-danger > .bp3-icon{
      color:#ff7373; }
.bp3-input{
  -webkit-appearance:none;
     -moz-appearance:none;
          appearance:none;
  background:#ffffff;
  border:none;
  border-radius:3px;
  -webkit-box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2);
          box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2);
  color:#182026;
  font-size:14px;
  font-weight:400;
  height:30px;
  line-height:30px;
  outline:none;
  padding:0 10px;
  -webkit-transition:-webkit-box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9);
  transition:-webkit-box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9);
  transition:box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9);
  transition:box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9), -webkit-box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9);
  vertical-align:middle; }
  .bp3-input::-webkit-input-placeholder{
    color:rgba(92, 112, 128, 0.6);
    opacity:1; }
  .bp3-input::-moz-placeholder{
    color:rgba(92, 112, 128, 0.6);
    opacity:1; }
  .bp3-input:-ms-input-placeholder{
    color:rgba(92, 112, 128, 0.6);
    opacity:1; }
  .bp3-input::-ms-input-placeholder{
    color:rgba(92, 112, 128, 0.6);
    opacity:1; }
  .bp3-input::placeholder{
    color:rgba(92, 112, 128, 0.6);
    opacity:1; }
  .bp3-input:focus, .bp3-input.bp3-active{
    -webkit-box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
  .bp3-input[type="search"], .bp3-input.bp3-round{
    border-radius:30px;
    -webkit-box-sizing:border-box;
            box-sizing:border-box;
    padding-left:10px; }
  .bp3-input[readonly]{
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.15);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.15); }
  .bp3-input:disabled, .bp3-input.bp3-disabled{
    background:rgba(206, 217, 224, 0.5);
    -webkit-box-shadow:none;
            box-shadow:none;
    color:rgba(92, 112, 128, 0.6);
    cursor:not-allowed;
    resize:none; }
  .bp3-input.bp3-large{
    font-size:16px;
    height:40px;
    line-height:40px; }
    .bp3-input.bp3-large[type="search"], .bp3-input.bp3-large.bp3-round{
      padding:0 15px; }
  .bp3-input.bp3-small{
    font-size:12px;
    height:24px;
    line-height:24px;
    padding-left:8px;
    padding-right:8px; }
    .bp3-input.bp3-small[type="search"], .bp3-input.bp3-small.bp3-round{
      padding:0 12px; }
  .bp3-input.bp3-fill{
    -webkit-box-flex:1;
        -ms-flex:1 1 auto;
            flex:1 1 auto;
    width:100%; }
  .bp3-dark .bp3-input{
    background:rgba(16, 22, 26, 0.3);
    -webkit-box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
    color:#f5f8fa; }
    .bp3-dark .bp3-input::-webkit-input-placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-input::-moz-placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-input:-ms-input-placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-input::-ms-input-placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-input::placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-input:focus{
      -webkit-box-shadow:0 0 0 1px #137cbd, 0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px #137cbd, 0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
    .bp3-dark .bp3-input[readonly]{
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4); }
    .bp3-dark .bp3-input:disabled, .bp3-dark .bp3-input.bp3-disabled{
      background:rgba(57, 75, 89, 0.5);
      -webkit-box-shadow:none;
              box-shadow:none;
      color:rgba(167, 182, 194, 0.6); }
  .bp3-input.bp3-intent-primary{
    -webkit-box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px #137cbd, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px #137cbd, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-input.bp3-intent-primary:focus{
      -webkit-box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-input.bp3-intent-primary[readonly]{
      -webkit-box-shadow:inset 0 0 0 1px #137cbd;
              box-shadow:inset 0 0 0 1px #137cbd; }
    .bp3-input.bp3-intent-primary:disabled, .bp3-input.bp3-intent-primary.bp3-disabled{
      -webkit-box-shadow:none;
              box-shadow:none; }
    .bp3-dark .bp3-input.bp3-intent-primary{
      -webkit-box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px #137cbd, inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px #137cbd, inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
      .bp3-dark .bp3-input.bp3-intent-primary:focus{
        -webkit-box-shadow:0 0 0 1px #137cbd, 0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
                box-shadow:0 0 0 1px #137cbd, 0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
      .bp3-dark .bp3-input.bp3-intent-primary[readonly]{
        -webkit-box-shadow:inset 0 0 0 1px #137cbd;
                box-shadow:inset 0 0 0 1px #137cbd; }
      .bp3-dark .bp3-input.bp3-intent-primary:disabled, .bp3-dark .bp3-input.bp3-intent-primary.bp3-disabled{
        -webkit-box-shadow:none;
                box-shadow:none; }
  .bp3-input.bp3-intent-success{
    -webkit-box-shadow:0 0 0 0 rgba(15, 153, 96, 0), 0 0 0 0 rgba(15, 153, 96, 0), inset 0 0 0 1px #0f9960, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 0 rgba(15, 153, 96, 0), 0 0 0 0 rgba(15, 153, 96, 0), inset 0 0 0 1px #0f9960, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-input.bp3-intent-success:focus{
      -webkit-box-shadow:0 0 0 1px #0f9960, 0 0 0 3px rgba(15, 153, 96, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px #0f9960, 0 0 0 3px rgba(15, 153, 96, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-input.bp3-intent-success[readonly]{
      -webkit-box-shadow:inset 0 0 0 1px #0f9960;
              box-shadow:inset 0 0 0 1px #0f9960; }
    .bp3-input.bp3-intent-success:disabled, .bp3-input.bp3-intent-success.bp3-disabled{
      -webkit-box-shadow:none;
              box-shadow:none; }
    .bp3-dark .bp3-input.bp3-intent-success{
      -webkit-box-shadow:0 0 0 0 rgba(15, 153, 96, 0), 0 0 0 0 rgba(15, 153, 96, 0), 0 0 0 0 rgba(15, 153, 96, 0), inset 0 0 0 1px #0f9960, inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 0 rgba(15, 153, 96, 0), 0 0 0 0 rgba(15, 153, 96, 0), 0 0 0 0 rgba(15, 153, 96, 0), inset 0 0 0 1px #0f9960, inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
      .bp3-dark .bp3-input.bp3-intent-success:focus{
        -webkit-box-shadow:0 0 0 1px #0f9960, 0 0 0 1px #0f9960, 0 0 0 3px rgba(15, 153, 96, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
                box-shadow:0 0 0 1px #0f9960, 0 0 0 1px #0f9960, 0 0 0 3px rgba(15, 153, 96, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
      .bp3-dark .bp3-input.bp3-intent-success[readonly]{
        -webkit-box-shadow:inset 0 0 0 1px #0f9960;
                box-shadow:inset 0 0 0 1px #0f9960; }
      .bp3-dark .bp3-input.bp3-intent-success:disabled, .bp3-dark .bp3-input.bp3-intent-success.bp3-disabled{
        -webkit-box-shadow:none;
                box-shadow:none; }
  .bp3-input.bp3-intent-warning{
    -webkit-box-shadow:0 0 0 0 rgba(217, 130, 43, 0), 0 0 0 0 rgba(217, 130, 43, 0), inset 0 0 0 1px #d9822b, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 0 rgba(217, 130, 43, 0), 0 0 0 0 rgba(217, 130, 43, 0), inset 0 0 0 1px #d9822b, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-input.bp3-intent-warning:focus{
      -webkit-box-shadow:0 0 0 1px #d9822b, 0 0 0 3px rgba(217, 130, 43, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px #d9822b, 0 0 0 3px rgba(217, 130, 43, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-input.bp3-intent-warning[readonly]{
      -webkit-box-shadow:inset 0 0 0 1px #d9822b;
              box-shadow:inset 0 0 0 1px #d9822b; }
    .bp3-input.bp3-intent-warning:disabled, .bp3-input.bp3-intent-warning.bp3-disabled{
      -webkit-box-shadow:none;
              box-shadow:none; }
    .bp3-dark .bp3-input.bp3-intent-warning{
      -webkit-box-shadow:0 0 0 0 rgba(217, 130, 43, 0), 0 0 0 0 rgba(217, 130, 43, 0), 0 0 0 0 rgba(217, 130, 43, 0), inset 0 0 0 1px #d9822b, inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 0 rgba(217, 130, 43, 0), 0 0 0 0 rgba(217, 130, 43, 0), 0 0 0 0 rgba(217, 130, 43, 0), inset 0 0 0 1px #d9822b, inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
      .bp3-dark .bp3-input.bp3-intent-warning:focus{
        -webkit-box-shadow:0 0 0 1px #d9822b, 0 0 0 1px #d9822b, 0 0 0 3px rgba(217, 130, 43, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
                box-shadow:0 0 0 1px #d9822b, 0 0 0 1px #d9822b, 0 0 0 3px rgba(217, 130, 43, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
      .bp3-dark .bp3-input.bp3-intent-warning[readonly]{
        -webkit-box-shadow:inset 0 0 0 1px #d9822b;
                box-shadow:inset 0 0 0 1px #d9822b; }
      .bp3-dark .bp3-input.bp3-intent-warning:disabled, .bp3-dark .bp3-input.bp3-intent-warning.bp3-disabled{
        -webkit-box-shadow:none;
                box-shadow:none; }
  .bp3-input.bp3-intent-danger{
    -webkit-box-shadow:0 0 0 0 rgba(219, 55, 55, 0), 0 0 0 0 rgba(219, 55, 55, 0), inset 0 0 0 1px #db3737, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 0 rgba(219, 55, 55, 0), 0 0 0 0 rgba(219, 55, 55, 0), inset 0 0 0 1px #db3737, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-input.bp3-intent-danger:focus{
      -webkit-box-shadow:0 0 0 1px #db3737, 0 0 0 3px rgba(219, 55, 55, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px #db3737, 0 0 0 3px rgba(219, 55, 55, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-input.bp3-intent-danger[readonly]{
      -webkit-box-shadow:inset 0 0 0 1px #db3737;
              box-shadow:inset 0 0 0 1px #db3737; }
    .bp3-input.bp3-intent-danger:disabled, .bp3-input.bp3-intent-danger.bp3-disabled{
      -webkit-box-shadow:none;
              box-shadow:none; }
    .bp3-dark .bp3-input.bp3-intent-danger{
      -webkit-box-shadow:0 0 0 0 rgba(219, 55, 55, 0), 0 0 0 0 rgba(219, 55, 55, 0), 0 0 0 0 rgba(219, 55, 55, 0), inset 0 0 0 1px #db3737, inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 0 rgba(219, 55, 55, 0), 0 0 0 0 rgba(219, 55, 55, 0), 0 0 0 0 rgba(219, 55, 55, 0), inset 0 0 0 1px #db3737, inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
      .bp3-dark .bp3-input.bp3-intent-danger:focus{
        -webkit-box-shadow:0 0 0 1px #db3737, 0 0 0 1px #db3737, 0 0 0 3px rgba(219, 55, 55, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
                box-shadow:0 0 0 1px #db3737, 0 0 0 1px #db3737, 0 0 0 3px rgba(219, 55, 55, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
      .bp3-dark .bp3-input.bp3-intent-danger[readonly]{
        -webkit-box-shadow:inset 0 0 0 1px #db3737;
                box-shadow:inset 0 0 0 1px #db3737; }
      .bp3-dark .bp3-input.bp3-intent-danger:disabled, .bp3-dark .bp3-input.bp3-intent-danger.bp3-disabled{
        -webkit-box-shadow:none;
                box-shadow:none; }
  .bp3-input::-ms-clear{
    display:none; }
textarea.bp3-input{
  max-width:100%;
  padding:10px; }
  textarea.bp3-input, textarea.bp3-input.bp3-large, textarea.bp3-input.bp3-small{
    height:auto;
    line-height:inherit; }
  textarea.bp3-input.bp3-small{
    padding:8px; }
  .bp3-dark textarea.bp3-input{
    background:rgba(16, 22, 26, 0.3);
    -webkit-box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
    color:#f5f8fa; }
    .bp3-dark textarea.bp3-input::-webkit-input-placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark textarea.bp3-input::-moz-placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark textarea.bp3-input:-ms-input-placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark textarea.bp3-input::-ms-input-placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark textarea.bp3-input::placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark textarea.bp3-input:focus{
      -webkit-box-shadow:0 0 0 1px #137cbd, 0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px #137cbd, 0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
    .bp3-dark textarea.bp3-input[readonly]{
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4); }
    .bp3-dark textarea.bp3-input:disabled, .bp3-dark textarea.bp3-input.bp3-disabled{
      background:rgba(57, 75, 89, 0.5);
      -webkit-box-shadow:none;
              box-shadow:none;
      color:rgba(167, 182, 194, 0.6); }
label.bp3-label{
  display:block;
  margin-bottom:15px;
  margin-top:0; }
  label.bp3-label .bp3-html-select,
  label.bp3-label .bp3-input,
  label.bp3-label .bp3-select,
  label.bp3-label .bp3-slider,
  label.bp3-label .bp3-popover-wrapper{
    display:block;
    margin-top:5px;
    text-transform:none; }
  label.bp3-label .bp3-button-group{
    margin-top:5px; }
  label.bp3-label .bp3-select select,
  label.bp3-label .bp3-html-select select{
    font-weight:400;
    vertical-align:top;
    width:100%; }
  label.bp3-label.bp3-disabled,
  label.bp3-label.bp3-disabled .bp3-text-muted{
    color:rgba(92, 112, 128, 0.6); }
  label.bp3-label.bp3-inline{
    line-height:30px; }
    label.bp3-label.bp3-inline .bp3-html-select,
    label.bp3-label.bp3-inline .bp3-input,
    label.bp3-label.bp3-inline .bp3-input-group,
    label.bp3-label.bp3-inline .bp3-select,
    label.bp3-label.bp3-inline .bp3-popover-wrapper{
      display:inline-block;
      margin:0 0 0 5px;
      vertical-align:top; }
    label.bp3-label.bp3-inline .bp3-button-group{
      margin:0 0 0 5px; }
    label.bp3-label.bp3-inline .bp3-input-group .bp3-input{
      margin-left:0; }
    label.bp3-label.bp3-inline.bp3-large{
      line-height:40px; }
  label.bp3-label:not(.bp3-inline) .bp3-popover-target{
    display:block; }
  .bp3-dark label.bp3-label{
    color:#f5f8fa; }
    .bp3-dark label.bp3-label.bp3-disabled,
    .bp3-dark label.bp3-label.bp3-disabled .bp3-text-muted{
      color:rgba(167, 182, 194, 0.6); }
.bp3-numeric-input .bp3-button-group.bp3-vertical > .bp3-button{
  -webkit-box-flex:1;
      -ms-flex:1 1 14px;
          flex:1 1 14px;
  min-height:0;
  padding:0;
  width:30px; }
  .bp3-numeric-input .bp3-button-group.bp3-vertical > .bp3-button:first-child{
    border-radius:0 3px 0 0; }
  .bp3-numeric-input .bp3-button-group.bp3-vertical > .bp3-button:last-child{
    border-radius:0 0 3px 0; }

.bp3-numeric-input .bp3-button-group.bp3-vertical:first-child > .bp3-button:first-child{
  border-radius:3px 0 0 0; }

.bp3-numeric-input .bp3-button-group.bp3-vertical:first-child > .bp3-button:last-child{
  border-radius:0 0 0 3px; }

.bp3-numeric-input.bp3-large .bp3-button-group.bp3-vertical > .bp3-button{
  width:40px; }

form{
  display:block; }
.bp3-html-select select,
.bp3-select select{
  display:-webkit-inline-box;
  display:-ms-inline-flexbox;
  display:inline-flex;
  -webkit-box-orient:horizontal;
  -webkit-box-direction:normal;
      -ms-flex-direction:row;
          flex-direction:row;
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  border:none;
  border-radius:3px;
  cursor:pointer;
  font-size:14px;
  -webkit-box-pack:center;
      -ms-flex-pack:center;
          justify-content:center;
  padding:5px 10px;
  text-align:left;
  vertical-align:middle;
  background-color:#f5f8fa;
  background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.8)), to(rgba(255, 255, 255, 0)));
  background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.8), rgba(255, 255, 255, 0));
  -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
          box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
  color:#182026;
  -moz-appearance:none;
  -webkit-appearance:none;
  border-radius:3px;
  height:30px;
  padding:0 25px 0 10px;
  width:100%; }
  .bp3-html-select select > *, .bp3-select select > *{
    -webkit-box-flex:0;
        -ms-flex-positive:0;
            flex-grow:0;
    -ms-flex-negative:0;
        flex-shrink:0; }
  .bp3-html-select select > .bp3-fill, .bp3-select select > .bp3-fill{
    -webkit-box-flex:1;
        -ms-flex-positive:1;
            flex-grow:1;
    -ms-flex-negative:1;
        flex-shrink:1; }
  .bp3-html-select select::before,
  .bp3-select select::before, .bp3-html-select select > *, .bp3-select select > *{
    margin-right:7px; }
  .bp3-html-select select:empty::before,
  .bp3-select select:empty::before,
  .bp3-html-select select > :last-child,
  .bp3-select select > :last-child{
    margin-right:0; }
  .bp3-html-select select:hover,
  .bp3-select select:hover{
    background-clip:padding-box;
    background-color:#ebf1f5;
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1); }
  .bp3-html-select select:active,
  .bp3-select select:active, .bp3-html-select select.bp3-active,
  .bp3-select select.bp3-active{
    background-color:#d8e1e8;
    background-image:none;
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 2px rgba(16, 22, 26, 0.2);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
  .bp3-html-select select:disabled,
  .bp3-select select:disabled, .bp3-html-select select.bp3-disabled,
  .bp3-select select.bp3-disabled{
    background-color:rgba(206, 217, 224, 0.5);
    background-image:none;
    -webkit-box-shadow:none;
            box-shadow:none;
    color:rgba(92, 112, 128, 0.6);
    cursor:not-allowed;
    outline:none; }
    .bp3-html-select select:disabled.bp3-active,
    .bp3-select select:disabled.bp3-active, .bp3-html-select select:disabled.bp3-active:hover,
    .bp3-select select:disabled.bp3-active:hover, .bp3-html-select select.bp3-disabled.bp3-active,
    .bp3-select select.bp3-disabled.bp3-active, .bp3-html-select select.bp3-disabled.bp3-active:hover,
    .bp3-select select.bp3-disabled.bp3-active:hover{
      background:rgba(206, 217, 224, 0.7); }

.bp3-html-select.bp3-minimal select,
.bp3-select.bp3-minimal select{
  background:none;
  -webkit-box-shadow:none;
          box-shadow:none; }
  .bp3-html-select.bp3-minimal select:hover,
  .bp3-select.bp3-minimal select:hover{
    background:rgba(167, 182, 194, 0.3);
    -webkit-box-shadow:none;
            box-shadow:none;
    color:#182026;
    text-decoration:none; }
  .bp3-html-select.bp3-minimal select:active,
  .bp3-select.bp3-minimal select:active, .bp3-html-select.bp3-minimal select.bp3-active,
  .bp3-select.bp3-minimal select.bp3-active{
    background:rgba(115, 134, 148, 0.3);
    -webkit-box-shadow:none;
            box-shadow:none;
    color:#182026; }
  .bp3-html-select.bp3-minimal select:disabled,
  .bp3-select.bp3-minimal select:disabled, .bp3-html-select.bp3-minimal select:disabled:hover,
  .bp3-select.bp3-minimal select:disabled:hover, .bp3-html-select.bp3-minimal select.bp3-disabled,
  .bp3-select.bp3-minimal select.bp3-disabled, .bp3-html-select.bp3-minimal select.bp3-disabled:hover,
  .bp3-select.bp3-minimal select.bp3-disabled:hover{
    background:none;
    color:rgba(92, 112, 128, 0.6);
    cursor:not-allowed; }
    .bp3-html-select.bp3-minimal select:disabled.bp3-active,
    .bp3-select.bp3-minimal select:disabled.bp3-active, .bp3-html-select.bp3-minimal select:disabled:hover.bp3-active,
    .bp3-select.bp3-minimal select:disabled:hover.bp3-active, .bp3-html-select.bp3-minimal select.bp3-disabled.bp3-active,
    .bp3-select.bp3-minimal select.bp3-disabled.bp3-active, .bp3-html-select.bp3-minimal select.bp3-disabled:hover.bp3-active,
    .bp3-select.bp3-minimal select.bp3-disabled:hover.bp3-active{
      background:rgba(115, 134, 148, 0.3); }
  .bp3-dark .bp3-html-select.bp3-minimal select, .bp3-html-select.bp3-minimal .bp3-dark select,
  .bp3-dark .bp3-select.bp3-minimal select, .bp3-select.bp3-minimal .bp3-dark select{
    background:none;
    -webkit-box-shadow:none;
            box-shadow:none;
    color:inherit; }
    .bp3-dark .bp3-html-select.bp3-minimal select:hover, .bp3-html-select.bp3-minimal .bp3-dark select:hover,
    .bp3-dark .bp3-select.bp3-minimal select:hover, .bp3-select.bp3-minimal .bp3-dark select:hover, .bp3-dark .bp3-html-select.bp3-minimal select:active, .bp3-html-select.bp3-minimal .bp3-dark select:active,
    .bp3-dark .bp3-select.bp3-minimal select:active, .bp3-select.bp3-minimal .bp3-dark select:active, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-active,
    .bp3-dark .bp3-select.bp3-minimal select.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-active{
      background:none;
      -webkit-box-shadow:none;
              box-shadow:none; }
    .bp3-dark .bp3-html-select.bp3-minimal select:hover, .bp3-html-select.bp3-minimal .bp3-dark select:hover,
    .bp3-dark .bp3-select.bp3-minimal select:hover, .bp3-select.bp3-minimal .bp3-dark select:hover{
      background:rgba(138, 155, 168, 0.15); }
    .bp3-dark .bp3-html-select.bp3-minimal select:active, .bp3-html-select.bp3-minimal .bp3-dark select:active,
    .bp3-dark .bp3-select.bp3-minimal select:active, .bp3-select.bp3-minimal .bp3-dark select:active, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-active,
    .bp3-dark .bp3-select.bp3-minimal select.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-active{
      background:rgba(138, 155, 168, 0.3);
      color:#f5f8fa; }
    .bp3-dark .bp3-html-select.bp3-minimal select:disabled, .bp3-html-select.bp3-minimal .bp3-dark select:disabled,
    .bp3-dark .bp3-select.bp3-minimal select:disabled, .bp3-select.bp3-minimal .bp3-dark select:disabled, .bp3-dark .bp3-html-select.bp3-minimal select:disabled:hover, .bp3-html-select.bp3-minimal .bp3-dark select:disabled:hover,
    .bp3-dark .bp3-select.bp3-minimal select:disabled:hover, .bp3-select.bp3-minimal .bp3-dark select:disabled:hover, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-disabled, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-disabled,
    .bp3-dark .bp3-select.bp3-minimal select.bp3-disabled, .bp3-select.bp3-minimal .bp3-dark select.bp3-disabled, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-disabled:hover, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-disabled:hover,
    .bp3-dark .bp3-select.bp3-minimal select.bp3-disabled:hover, .bp3-select.bp3-minimal .bp3-dark select.bp3-disabled:hover{
      background:none;
      color:rgba(167, 182, 194, 0.6);
      cursor:not-allowed; }
      .bp3-dark .bp3-html-select.bp3-minimal select:disabled.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select:disabled.bp3-active,
      .bp3-dark .bp3-select.bp3-minimal select:disabled.bp3-active, .bp3-select.bp3-minimal .bp3-dark select:disabled.bp3-active, .bp3-dark .bp3-html-select.bp3-minimal select:disabled:hover.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select:disabled:hover.bp3-active,
      .bp3-dark .bp3-select.bp3-minimal select:disabled:hover.bp3-active, .bp3-select.bp3-minimal .bp3-dark select:disabled:hover.bp3-active, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-disabled.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-disabled.bp3-active,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-disabled.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-disabled.bp3-active, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-disabled:hover.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-disabled:hover.bp3-active,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-disabled:hover.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-disabled:hover.bp3-active{
        background:rgba(138, 155, 168, 0.3); }
  .bp3-html-select.bp3-minimal select.bp3-intent-primary,
  .bp3-select.bp3-minimal select.bp3-intent-primary{
    color:#106ba3; }
    .bp3-html-select.bp3-minimal select.bp3-intent-primary:hover,
    .bp3-select.bp3-minimal select.bp3-intent-primary:hover, .bp3-html-select.bp3-minimal select.bp3-intent-primary:active,
    .bp3-select.bp3-minimal select.bp3-intent-primary:active, .bp3-html-select.bp3-minimal select.bp3-intent-primary.bp3-active,
    .bp3-select.bp3-minimal select.bp3-intent-primary.bp3-active{
      background:none;
      -webkit-box-shadow:none;
              box-shadow:none;
      color:#106ba3; }
    .bp3-html-select.bp3-minimal select.bp3-intent-primary:hover,
    .bp3-select.bp3-minimal select.bp3-intent-primary:hover{
      background:rgba(19, 124, 189, 0.15);
      color:#106ba3; }
    .bp3-html-select.bp3-minimal select.bp3-intent-primary:active,
    .bp3-select.bp3-minimal select.bp3-intent-primary:active, .bp3-html-select.bp3-minimal select.bp3-intent-primary.bp3-active,
    .bp3-select.bp3-minimal select.bp3-intent-primary.bp3-active{
      background:rgba(19, 124, 189, 0.3);
      color:#106ba3; }
    .bp3-html-select.bp3-minimal select.bp3-intent-primary:disabled,
    .bp3-select.bp3-minimal select.bp3-intent-primary:disabled, .bp3-html-select.bp3-minimal select.bp3-intent-primary.bp3-disabled,
    .bp3-select.bp3-minimal select.bp3-intent-primary.bp3-disabled{
      background:none;
      color:rgba(16, 107, 163, 0.5); }
      .bp3-html-select.bp3-minimal select.bp3-intent-primary:disabled.bp3-active,
      .bp3-select.bp3-minimal select.bp3-intent-primary:disabled.bp3-active, .bp3-html-select.bp3-minimal select.bp3-intent-primary.bp3-disabled.bp3-active,
      .bp3-select.bp3-minimal select.bp3-intent-primary.bp3-disabled.bp3-active{
        background:rgba(19, 124, 189, 0.3); }
    .bp3-html-select.bp3-minimal select.bp3-intent-primary .bp3-button-spinner .bp3-spinner-head, .bp3-select.bp3-minimal select.bp3-intent-primary .bp3-button-spinner .bp3-spinner-head{
      stroke:#106ba3; }
    .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-primary, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-primary,
    .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-primary, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-primary{
      color:#48aff0; }
      .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-primary:hover, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-primary:hover,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-primary:hover, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-primary:hover{
        background:rgba(19, 124, 189, 0.2);
        color:#48aff0; }
      .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-primary:active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-primary:active,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-primary:active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-primary:active, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-primary.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-primary.bp3-active,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-primary.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-primary.bp3-active{
        background:rgba(19, 124, 189, 0.3);
        color:#48aff0; }
      .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-primary:disabled, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-primary:disabled,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-primary:disabled, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-primary:disabled, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-primary.bp3-disabled, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-primary.bp3-disabled,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-primary.bp3-disabled, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-primary.bp3-disabled{
        background:none;
        color:rgba(72, 175, 240, 0.5); }
        .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-primary:disabled.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-primary:disabled.bp3-active,
        .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-primary:disabled.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-primary:disabled.bp3-active, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-primary.bp3-disabled.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-primary.bp3-disabled.bp3-active,
        .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-primary.bp3-disabled.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-primary.bp3-disabled.bp3-active{
          background:rgba(19, 124, 189, 0.3); }
  .bp3-html-select.bp3-minimal select.bp3-intent-success,
  .bp3-select.bp3-minimal select.bp3-intent-success{
    color:#0d8050; }
    .bp3-html-select.bp3-minimal select.bp3-intent-success:hover,
    .bp3-select.bp3-minimal select.bp3-intent-success:hover, .bp3-html-select.bp3-minimal select.bp3-intent-success:active,
    .bp3-select.bp3-minimal select.bp3-intent-success:active, .bp3-html-select.bp3-minimal select.bp3-intent-success.bp3-active,
    .bp3-select.bp3-minimal select.bp3-intent-success.bp3-active{
      background:none;
      -webkit-box-shadow:none;
              box-shadow:none;
      color:#0d8050; }
    .bp3-html-select.bp3-minimal select.bp3-intent-success:hover,
    .bp3-select.bp3-minimal select.bp3-intent-success:hover{
      background:rgba(15, 153, 96, 0.15);
      color:#0d8050; }
    .bp3-html-select.bp3-minimal select.bp3-intent-success:active,
    .bp3-select.bp3-minimal select.bp3-intent-success:active, .bp3-html-select.bp3-minimal select.bp3-intent-success.bp3-active,
    .bp3-select.bp3-minimal select.bp3-intent-success.bp3-active{
      background:rgba(15, 153, 96, 0.3);
      color:#0d8050; }
    .bp3-html-select.bp3-minimal select.bp3-intent-success:disabled,
    .bp3-select.bp3-minimal select.bp3-intent-success:disabled, .bp3-html-select.bp3-minimal select.bp3-intent-success.bp3-disabled,
    .bp3-select.bp3-minimal select.bp3-intent-success.bp3-disabled{
      background:none;
      color:rgba(13, 128, 80, 0.5); }
      .bp3-html-select.bp3-minimal select.bp3-intent-success:disabled.bp3-active,
      .bp3-select.bp3-minimal select.bp3-intent-success:disabled.bp3-active, .bp3-html-select.bp3-minimal select.bp3-intent-success.bp3-disabled.bp3-active,
      .bp3-select.bp3-minimal select.bp3-intent-success.bp3-disabled.bp3-active{
        background:rgba(15, 153, 96, 0.3); }
    .bp3-html-select.bp3-minimal select.bp3-intent-success .bp3-button-spinner .bp3-spinner-head, .bp3-select.bp3-minimal select.bp3-intent-success .bp3-button-spinner .bp3-spinner-head{
      stroke:#0d8050; }
    .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-success, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-success,
    .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-success, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-success{
      color:#3dcc91; }
      .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-success:hover, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-success:hover,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-success:hover, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-success:hover{
        background:rgba(15, 153, 96, 0.2);
        color:#3dcc91; }
      .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-success:active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-success:active,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-success:active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-success:active, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-success.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-success.bp3-active,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-success.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-success.bp3-active{
        background:rgba(15, 153, 96, 0.3);
        color:#3dcc91; }
      .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-success:disabled, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-success:disabled,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-success:disabled, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-success:disabled, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-success.bp3-disabled, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-success.bp3-disabled,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-success.bp3-disabled, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-success.bp3-disabled{
        background:none;
        color:rgba(61, 204, 145, 0.5); }
        .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-success:disabled.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-success:disabled.bp3-active,
        .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-success:disabled.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-success:disabled.bp3-active, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-success.bp3-disabled.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-success.bp3-disabled.bp3-active,
        .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-success.bp3-disabled.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-success.bp3-disabled.bp3-active{
          background:rgba(15, 153, 96, 0.3); }
  .bp3-html-select.bp3-minimal select.bp3-intent-warning,
  .bp3-select.bp3-minimal select.bp3-intent-warning{
    color:#bf7326; }
    .bp3-html-select.bp3-minimal select.bp3-intent-warning:hover,
    .bp3-select.bp3-minimal select.bp3-intent-warning:hover, .bp3-html-select.bp3-minimal select.bp3-intent-warning:active,
    .bp3-select.bp3-minimal select.bp3-intent-warning:active, .bp3-html-select.bp3-minimal select.bp3-intent-warning.bp3-active,
    .bp3-select.bp3-minimal select.bp3-intent-warning.bp3-active{
      background:none;
      -webkit-box-shadow:none;
              box-shadow:none;
      color:#bf7326; }
    .bp3-html-select.bp3-minimal select.bp3-intent-warning:hover,
    .bp3-select.bp3-minimal select.bp3-intent-warning:hover{
      background:rgba(217, 130, 43, 0.15);
      color:#bf7326; }
    .bp3-html-select.bp3-minimal select.bp3-intent-warning:active,
    .bp3-select.bp3-minimal select.bp3-intent-warning:active, .bp3-html-select.bp3-minimal select.bp3-intent-warning.bp3-active,
    .bp3-select.bp3-minimal select.bp3-intent-warning.bp3-active{
      background:rgba(217, 130, 43, 0.3);
      color:#bf7326; }
    .bp3-html-select.bp3-minimal select.bp3-intent-warning:disabled,
    .bp3-select.bp3-minimal select.bp3-intent-warning:disabled, .bp3-html-select.bp3-minimal select.bp3-intent-warning.bp3-disabled,
    .bp3-select.bp3-minimal select.bp3-intent-warning.bp3-disabled{
      background:none;
      color:rgba(191, 115, 38, 0.5); }
      .bp3-html-select.bp3-minimal select.bp3-intent-warning:disabled.bp3-active,
      .bp3-select.bp3-minimal select.bp3-intent-warning:disabled.bp3-active, .bp3-html-select.bp3-minimal select.bp3-intent-warning.bp3-disabled.bp3-active,
      .bp3-select.bp3-minimal select.bp3-intent-warning.bp3-disabled.bp3-active{
        background:rgba(217, 130, 43, 0.3); }
    .bp3-html-select.bp3-minimal select.bp3-intent-warning .bp3-button-spinner .bp3-spinner-head, .bp3-select.bp3-minimal select.bp3-intent-warning .bp3-button-spinner .bp3-spinner-head{
      stroke:#bf7326; }
    .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-warning, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-warning,
    .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-warning, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-warning{
      color:#ffb366; }
      .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-warning:hover, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-warning:hover,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-warning:hover, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-warning:hover{
        background:rgba(217, 130, 43, 0.2);
        color:#ffb366; }
      .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-warning:active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-warning:active,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-warning:active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-warning:active, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-warning.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-warning.bp3-active,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-warning.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-warning.bp3-active{
        background:rgba(217, 130, 43, 0.3);
        color:#ffb366; }
      .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-warning:disabled, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-warning:disabled,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-warning:disabled, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-warning:disabled, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-warning.bp3-disabled, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-warning.bp3-disabled,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-warning.bp3-disabled, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-warning.bp3-disabled{
        background:none;
        color:rgba(255, 179, 102, 0.5); }
        .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-warning:disabled.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-warning:disabled.bp3-active,
        .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-warning:disabled.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-warning:disabled.bp3-active, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-warning.bp3-disabled.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-warning.bp3-disabled.bp3-active,
        .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-warning.bp3-disabled.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-warning.bp3-disabled.bp3-active{
          background:rgba(217, 130, 43, 0.3); }
  .bp3-html-select.bp3-minimal select.bp3-intent-danger,
  .bp3-select.bp3-minimal select.bp3-intent-danger{
    color:#c23030; }
    .bp3-html-select.bp3-minimal select.bp3-intent-danger:hover,
    .bp3-select.bp3-minimal select.bp3-intent-danger:hover, .bp3-html-select.bp3-minimal select.bp3-intent-danger:active,
    .bp3-select.bp3-minimal select.bp3-intent-danger:active, .bp3-html-select.bp3-minimal select.bp3-intent-danger.bp3-active,
    .bp3-select.bp3-minimal select.bp3-intent-danger.bp3-active{
      background:none;
      -webkit-box-shadow:none;
              box-shadow:none;
      color:#c23030; }
    .bp3-html-select.bp3-minimal select.bp3-intent-danger:hover,
    .bp3-select.bp3-minimal select.bp3-intent-danger:hover{
      background:rgba(219, 55, 55, 0.15);
      color:#c23030; }
    .bp3-html-select.bp3-minimal select.bp3-intent-danger:active,
    .bp3-select.bp3-minimal select.bp3-intent-danger:active, .bp3-html-select.bp3-minimal select.bp3-intent-danger.bp3-active,
    .bp3-select.bp3-minimal select.bp3-intent-danger.bp3-active{
      background:rgba(219, 55, 55, 0.3);
      color:#c23030; }
    .bp3-html-select.bp3-minimal select.bp3-intent-danger:disabled,
    .bp3-select.bp3-minimal select.bp3-intent-danger:disabled, .bp3-html-select.bp3-minimal select.bp3-intent-danger.bp3-disabled,
    .bp3-select.bp3-minimal select.bp3-intent-danger.bp3-disabled{
      background:none;
      color:rgba(194, 48, 48, 0.5); }
      .bp3-html-select.bp3-minimal select.bp3-intent-danger:disabled.bp3-active,
      .bp3-select.bp3-minimal select.bp3-intent-danger:disabled.bp3-active, .bp3-html-select.bp3-minimal select.bp3-intent-danger.bp3-disabled.bp3-active,
      .bp3-select.bp3-minimal select.bp3-intent-danger.bp3-disabled.bp3-active{
        background:rgba(219, 55, 55, 0.3); }
    .bp3-html-select.bp3-minimal select.bp3-intent-danger .bp3-button-spinner .bp3-spinner-head, .bp3-select.bp3-minimal select.bp3-intent-danger .bp3-button-spinner .bp3-spinner-head{
      stroke:#c23030; }
    .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-danger, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-danger,
    .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-danger, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-danger{
      color:#ff7373; }
      .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-danger:hover, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-danger:hover,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-danger:hover, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-danger:hover{
        background:rgba(219, 55, 55, 0.2);
        color:#ff7373; }
      .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-danger:active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-danger:active,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-danger:active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-danger:active, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-danger.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-danger.bp3-active,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-danger.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-danger.bp3-active{
        background:rgba(219, 55, 55, 0.3);
        color:#ff7373; }
      .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-danger:disabled, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-danger:disabled,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-danger:disabled, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-danger:disabled, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-danger.bp3-disabled, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-danger.bp3-disabled,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-danger.bp3-disabled, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-danger.bp3-disabled{
        background:none;
        color:rgba(255, 115, 115, 0.5); }
        .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-danger:disabled.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-danger:disabled.bp3-active,
        .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-danger:disabled.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-danger:disabled.bp3-active, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-danger.bp3-disabled.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-danger.bp3-disabled.bp3-active,
        .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-danger.bp3-disabled.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-danger.bp3-disabled.bp3-active{
          background:rgba(219, 55, 55, 0.3); }

.bp3-html-select.bp3-large select,
.bp3-select.bp3-large select{
  font-size:16px;
  height:40px;
  padding-right:35px; }

.bp3-dark .bp3-html-select select, .bp3-dark .bp3-select select{
  background-color:#394b59;
  background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.05)), to(rgba(255, 255, 255, 0)));
  background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0));
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
  color:#f5f8fa; }
  .bp3-dark .bp3-html-select select:hover, .bp3-dark .bp3-select select:hover, .bp3-dark .bp3-html-select select:active, .bp3-dark .bp3-select select:active, .bp3-dark .bp3-html-select select.bp3-active, .bp3-dark .bp3-select select.bp3-active{
    color:#f5f8fa; }
  .bp3-dark .bp3-html-select select:hover, .bp3-dark .bp3-select select:hover{
    background-color:#30404d;
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4); }
  .bp3-dark .bp3-html-select select:active, .bp3-dark .bp3-select select:active, .bp3-dark .bp3-html-select select.bp3-active, .bp3-dark .bp3-select select.bp3-active{
    background-color:#202b33;
    background-image:none;
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.6), inset 0 1px 2px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.6), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
  .bp3-dark .bp3-html-select select:disabled, .bp3-dark .bp3-select select:disabled, .bp3-dark .bp3-html-select select.bp3-disabled, .bp3-dark .bp3-select select.bp3-disabled{
    background-color:rgba(57, 75, 89, 0.5);
    background-image:none;
    -webkit-box-shadow:none;
            box-shadow:none;
    color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-html-select select:disabled.bp3-active, .bp3-dark .bp3-select select:disabled.bp3-active, .bp3-dark .bp3-html-select select.bp3-disabled.bp3-active, .bp3-dark .bp3-select select.bp3-disabled.bp3-active{
      background:rgba(57, 75, 89, 0.7); }
  .bp3-dark .bp3-html-select select .bp3-button-spinner .bp3-spinner-head, .bp3-dark .bp3-select select .bp3-button-spinner .bp3-spinner-head{
    background:rgba(16, 22, 26, 0.5);
    stroke:#8a9ba8; }

.bp3-html-select select:disabled,
.bp3-select select:disabled{
  background-color:rgba(206, 217, 224, 0.5);
  -webkit-box-shadow:none;
          box-shadow:none;
  color:rgba(92, 112, 128, 0.6);
  cursor:not-allowed; }

.bp3-html-select .bp3-icon,
.bp3-select .bp3-icon, .bp3-select::after{
  color:#5c7080;
  pointer-events:none;
  position:absolute;
  right:7px;
  top:7px; }
  .bp3-html-select .bp3-disabled.bp3-icon,
  .bp3-select .bp3-disabled.bp3-icon, .bp3-disabled.bp3-select::after{
    color:rgba(92, 112, 128, 0.6); }
.bp3-html-select,
.bp3-select{
  display:inline-block;
  letter-spacing:normal;
  position:relative;
  vertical-align:middle; }
  .bp3-html-select select::-ms-expand,
  .bp3-select select::-ms-expand{
    display:none; }
  .bp3-html-select .bp3-icon,
  .bp3-select .bp3-icon{
    color:#5c7080; }
    .bp3-html-select .bp3-icon:hover,
    .bp3-select .bp3-icon:hover{
      color:#182026; }
    .bp3-dark .bp3-html-select .bp3-icon, .bp3-dark
    .bp3-select .bp3-icon{
      color:#a7b6c2; }
      .bp3-dark .bp3-html-select .bp3-icon:hover, .bp3-dark
      .bp3-select .bp3-icon:hover{
        color:#f5f8fa; }
  .bp3-html-select.bp3-large::after,
  .bp3-html-select.bp3-large .bp3-icon,
  .bp3-select.bp3-large::after,
  .bp3-select.bp3-large .bp3-icon{
    right:12px;
    top:12px; }
  .bp3-html-select.bp3-fill,
  .bp3-html-select.bp3-fill select,
  .bp3-select.bp3-fill,
  .bp3-select.bp3-fill select{
    width:100%; }
  .bp3-dark .bp3-html-select option, .bp3-dark
  .bp3-select option{
    background-color:#30404d;
    color:#f5f8fa; }
  .bp3-dark .bp3-html-select option:disabled, .bp3-dark
  .bp3-select option:disabled{
    color:rgba(167, 182, 194, 0.6); }
  .bp3-dark .bp3-html-select::after, .bp3-dark
  .bp3-select::after{
    color:#a7b6c2; }

.bp3-select::after{
  font-family:"Icons16", sans-serif;
  font-size:16px;
  font-style:normal;
  font-weight:400;
  line-height:1;
  -moz-osx-font-smoothing:grayscale;
  -webkit-font-smoothing:antialiased;
  content:""; }
.bp3-running-text table, table.bp3-html-table{
  border-spacing:0;
  font-size:14px; }
  .bp3-running-text table th, table.bp3-html-table th,
  .bp3-running-text table td,
  table.bp3-html-table td{
    padding:11px;
    text-align:left;
    vertical-align:top; }
  .bp3-running-text table th, table.bp3-html-table th{
    color:#182026;
    font-weight:600; }
  
  .bp3-running-text table td,
  table.bp3-html-table td{
    color:#182026; }
  .bp3-running-text table tbody tr:first-child th, table.bp3-html-table tbody tr:first-child th,
  .bp3-running-text table tbody tr:first-child td,
  table.bp3-html-table tbody tr:first-child td,
  .bp3-running-text table tfoot tr:first-child th,
  table.bp3-html-table tfoot tr:first-child th,
  .bp3-running-text table tfoot tr:first-child td,
  table.bp3-html-table tfoot tr:first-child td{
    -webkit-box-shadow:inset 0 1px 0 0 rgba(16, 22, 26, 0.15);
            box-shadow:inset 0 1px 0 0 rgba(16, 22, 26, 0.15); }
  .bp3-dark .bp3-running-text table th, .bp3-running-text .bp3-dark table th, .bp3-dark table.bp3-html-table th{
    color:#f5f8fa; }
  .bp3-dark .bp3-running-text table td, .bp3-running-text .bp3-dark table td, .bp3-dark table.bp3-html-table td{
    color:#f5f8fa; }
  .bp3-dark .bp3-running-text table tbody tr:first-child th, .bp3-running-text .bp3-dark table tbody tr:first-child th, .bp3-dark table.bp3-html-table tbody tr:first-child th,
  .bp3-dark .bp3-running-text table tbody tr:first-child td,
  .bp3-running-text .bp3-dark table tbody tr:first-child td,
  .bp3-dark table.bp3-html-table tbody tr:first-child td,
  .bp3-dark .bp3-running-text table tfoot tr:first-child th,
  .bp3-running-text .bp3-dark table tfoot tr:first-child th,
  .bp3-dark table.bp3-html-table tfoot tr:first-child th,
  .bp3-dark .bp3-running-text table tfoot tr:first-child td,
  .bp3-running-text .bp3-dark table tfoot tr:first-child td,
  .bp3-dark table.bp3-html-table tfoot tr:first-child td{
    -webkit-box-shadow:inset 0 1px 0 0 rgba(255, 255, 255, 0.15);
            box-shadow:inset 0 1px 0 0 rgba(255, 255, 255, 0.15); }

table.bp3-html-table.bp3-html-table-condensed th,
table.bp3-html-table.bp3-html-table-condensed td, table.bp3-html-table.bp3-small th,
table.bp3-html-table.bp3-small td{
  padding-bottom:6px;
  padding-top:6px; }

table.bp3-html-table.bp3-html-table-striped tbody tr:nth-child(odd) td{
  background:rgba(191, 204, 214, 0.15); }

table.bp3-html-table.bp3-html-table-bordered th:not(:first-child){
  -webkit-box-shadow:inset 1px 0 0 0 rgba(16, 22, 26, 0.15);
          box-shadow:inset 1px 0 0 0 rgba(16, 22, 26, 0.15); }

table.bp3-html-table.bp3-html-table-bordered tbody tr td,
table.bp3-html-table.bp3-html-table-bordered tfoot tr td{
  -webkit-box-shadow:inset 0 1px 0 0 rgba(16, 22, 26, 0.15);
          box-shadow:inset 0 1px 0 0 rgba(16, 22, 26, 0.15); }
  table.bp3-html-table.bp3-html-table-bordered tbody tr td:not(:first-child),
  table.bp3-html-table.bp3-html-table-bordered tfoot tr td:not(:first-child){
    -webkit-box-shadow:inset 1px 1px 0 0 rgba(16, 22, 26, 0.15);
            box-shadow:inset 1px 1px 0 0 rgba(16, 22, 26, 0.15); }

table.bp3-html-table.bp3-html-table-bordered.bp3-html-table-striped tbody tr:not(:first-child) td{
  -webkit-box-shadow:none;
          box-shadow:none; }
  table.bp3-html-table.bp3-html-table-bordered.bp3-html-table-striped tbody tr:not(:first-child) td:not(:first-child){
    -webkit-box-shadow:inset 1px 0 0 0 rgba(16, 22, 26, 0.15);
            box-shadow:inset 1px 0 0 0 rgba(16, 22, 26, 0.15); }

table.bp3-html-table.bp3-interactive tbody tr:hover td{
  background-color:rgba(191, 204, 214, 0.3);
  cursor:pointer; }

table.bp3-html-table.bp3-interactive tbody tr:active td{
  background-color:rgba(191, 204, 214, 0.4); }

.bp3-dark table.bp3-html-table{ }
  .bp3-dark table.bp3-html-table.bp3-html-table-striped tbody tr:nth-child(odd) td{
    background:rgba(92, 112, 128, 0.15); }
  .bp3-dark table.bp3-html-table.bp3-html-table-bordered th:not(:first-child){
    -webkit-box-shadow:inset 1px 0 0 0 rgba(255, 255, 255, 0.15);
            box-shadow:inset 1px 0 0 0 rgba(255, 255, 255, 0.15); }
  .bp3-dark table.bp3-html-table.bp3-html-table-bordered tbody tr td,
  .bp3-dark table.bp3-html-table.bp3-html-table-bordered tfoot tr td{
    -webkit-box-shadow:inset 0 1px 0 0 rgba(255, 255, 255, 0.15);
            box-shadow:inset 0 1px 0 0 rgba(255, 255, 255, 0.15); }
    .bp3-dark table.bp3-html-table.bp3-html-table-bordered tbody tr td:not(:first-child),
    .bp3-dark table.bp3-html-table.bp3-html-table-bordered tfoot tr td:not(:first-child){
      -webkit-box-shadow:inset 1px 1px 0 0 rgba(255, 255, 255, 0.15);
              box-shadow:inset 1px 1px 0 0 rgba(255, 255, 255, 0.15); }
  .bp3-dark table.bp3-html-table.bp3-html-table-bordered.bp3-html-table-striped tbody tr:not(:first-child) td{
    -webkit-box-shadow:inset 1px 0 0 0 rgba(255, 255, 255, 0.15);
            box-shadow:inset 1px 0 0 0 rgba(255, 255, 255, 0.15); }
    .bp3-dark table.bp3-html-table.bp3-html-table-bordered.bp3-html-table-striped tbody tr:not(:first-child) td:first-child{
      -webkit-box-shadow:none;
              box-shadow:none; }
  .bp3-dark table.bp3-html-table.bp3-interactive tbody tr:hover td{
    background-color:rgba(92, 112, 128, 0.3);
    cursor:pointer; }
  .bp3-dark table.bp3-html-table.bp3-interactive tbody tr:active td{
    background-color:rgba(92, 112, 128, 0.4); }

.bp3-key-combo{
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-orient:horizontal;
  -webkit-box-direction:normal;
      -ms-flex-direction:row;
          flex-direction:row;
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center; }
  .bp3-key-combo > *{
    -webkit-box-flex:0;
        -ms-flex-positive:0;
            flex-grow:0;
    -ms-flex-negative:0;
        flex-shrink:0; }
  .bp3-key-combo > .bp3-fill{
    -webkit-box-flex:1;
        -ms-flex-positive:1;
            flex-grow:1;
    -ms-flex-negative:1;
        flex-shrink:1; }
  .bp3-key-combo::before,
  .bp3-key-combo > *{
    margin-right:5px; }
  .bp3-key-combo:empty::before,
  .bp3-key-combo > :last-child{
    margin-right:0; }

.bp3-hotkey-dialog{
  padding-bottom:0;
  top:40px; }
  .bp3-hotkey-dialog .bp3-dialog-body{
    margin:0;
    padding:0; }
  .bp3-hotkey-dialog .bp3-hotkey-label{
    -webkit-box-flex:1;
        -ms-flex-positive:1;
            flex-grow:1; }

.bp3-hotkey-column{
  margin:auto;
  max-height:80vh;
  overflow-y:auto;
  padding:30px; }
  .bp3-hotkey-column .bp3-heading{
    margin-bottom:20px; }
    .bp3-hotkey-column .bp3-heading:not(:first-child){
      margin-top:40px; }

.bp3-hotkey{
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-pack:justify;
      -ms-flex-pack:justify;
          justify-content:space-between;
  margin-left:0;
  margin-right:0; }
  .bp3-hotkey:not(:last-child){
    margin-bottom:10px; }
.bp3-icon{
  display:inline-block;
  -webkit-box-flex:0;
      -ms-flex:0 0 auto;
          flex:0 0 auto;
  vertical-align:text-bottom; }
  .bp3-icon:not(:empty)::before{
    content:"" !important;
    content:unset !important; }
  .bp3-icon > svg{
    display:block; }
    .bp3-icon > svg:not([fill]){
      fill:currentColor; }

.bp3-icon.bp3-intent-primary, .bp3-icon-standard.bp3-intent-primary, .bp3-icon-large.bp3-intent-primary{
  color:#106ba3; }
  .bp3-dark .bp3-icon.bp3-intent-primary, .bp3-dark .bp3-icon-standard.bp3-intent-primary, .bp3-dark .bp3-icon-large.bp3-intent-primary{
    color:#48aff0; }

.bp3-icon.bp3-intent-success, .bp3-icon-standard.bp3-intent-success, .bp3-icon-large.bp3-intent-success{
  color:#0d8050; }
  .bp3-dark .bp3-icon.bp3-intent-success, .bp3-dark .bp3-icon-standard.bp3-intent-success, .bp3-dark .bp3-icon-large.bp3-intent-success{
    color:#3dcc91; }

.bp3-icon.bp3-intent-warning, .bp3-icon-standard.bp3-intent-warning, .bp3-icon-large.bp3-intent-warning{
  color:#bf7326; }
  .bp3-dark .bp3-icon.bp3-intent-warning, .bp3-dark .bp3-icon-standard.bp3-intent-warning, .bp3-dark .bp3-icon-large.bp3-intent-warning{
    color:#ffb366; }

.bp3-icon.bp3-intent-danger, .bp3-icon-standard.bp3-intent-danger, .bp3-icon-large.bp3-intent-danger{
  color:#c23030; }
  .bp3-dark .bp3-icon.bp3-intent-danger, .bp3-dark .bp3-icon-standard.bp3-intent-danger, .bp3-dark .bp3-icon-large.bp3-intent-danger{
    color:#ff7373; }

span.bp3-icon-standard{
  font-family:"Icons16", sans-serif;
  font-size:16px;
  font-style:normal;
  font-weight:400;
  line-height:1;
  -moz-osx-font-smoothing:grayscale;
  -webkit-font-smoothing:antialiased;
  display:inline-block; }

span.bp3-icon-large{
  font-family:"Icons20", sans-serif;
  font-size:20px;
  font-style:normal;
  font-weight:400;
  line-height:1;
  -moz-osx-font-smoothing:grayscale;
  -webkit-font-smoothing:antialiased;
  display:inline-block; }

span.bp3-icon:empty{
  font-family:"Icons20";
  font-size:inherit;
  font-style:normal;
  font-weight:400;
  line-height:1; }
  span.bp3-icon:empty::before{
    -moz-osx-font-smoothing:grayscale;
    -webkit-font-smoothing:antialiased; }

.bp3-icon-add::before{
  content:""; }

.bp3-icon-add-column-left::before{
  content:""; }

.bp3-icon-add-column-right::before{
  content:""; }

.bp3-icon-add-row-bottom::before{
  content:""; }

.bp3-icon-add-row-top::before{
  content:""; }

.bp3-icon-add-to-artifact::before{
  content:""; }

.bp3-icon-add-to-folder::before{
  content:""; }

.bp3-icon-airplane::before{
  content:""; }

.bp3-icon-align-center::before{
  content:""; }

.bp3-icon-align-justify::before{
  content:""; }

.bp3-icon-align-left::before{
  content:""; }

.bp3-icon-align-right::before{
  content:""; }

.bp3-icon-alignment-bottom::before{
  content:""; }

.bp3-icon-alignment-horizontal-center::before{
  content:""; }

.bp3-icon-alignment-left::before{
  content:""; }

.bp3-icon-alignment-right::before{
  content:""; }

.bp3-icon-alignment-top::before{
  content:""; }

.bp3-icon-alignment-vertical-center::before{
  content:""; }

.bp3-icon-annotation::before{
  content:""; }

.bp3-icon-application::before{
  content:""; }

.bp3-icon-applications::before{
  content:""; }

.bp3-icon-archive::before{
  content:""; }

.bp3-icon-arrow-bottom-left::before{
  content:"↙"; }

.bp3-icon-arrow-bottom-right::before{
  content:"↘"; }

.bp3-icon-arrow-down::before{
  content:"↓"; }

.bp3-icon-arrow-left::before{
  content:"←"; }

.bp3-icon-arrow-right::before{
  content:"→"; }

.bp3-icon-arrow-top-left::before{
  content:"↖"; }

.bp3-icon-arrow-top-right::before{
  content:"↗"; }

.bp3-icon-arrow-up::before{
  content:"↑"; }

.bp3-icon-arrows-horizontal::before{
  content:"↔"; }

.bp3-icon-arrows-vertical::before{
  content:"↕"; }

.bp3-icon-asterisk::before{
  content:"*"; }

.bp3-icon-automatic-updates::before{
  content:""; }

.bp3-icon-badge::before{
  content:""; }

.bp3-icon-ban-circle::before{
  content:""; }

.bp3-icon-bank-account::before{
  content:""; }

.bp3-icon-barcode::before{
  content:""; }

.bp3-icon-blank::before{
  content:""; }

.bp3-icon-blocked-person::before{
  content:""; }

.bp3-icon-bold::before{
  content:""; }

.bp3-icon-book::before{
  content:""; }

.bp3-icon-bookmark::before{
  content:""; }

.bp3-icon-box::before{
  content:""; }

.bp3-icon-briefcase::before{
  content:""; }

.bp3-icon-bring-data::before{
  content:""; }

.bp3-icon-build::before{
  content:""; }

.bp3-icon-calculator::before{
  content:""; }

.bp3-icon-calendar::before{
  content:""; }

.bp3-icon-camera::before{
  content:""; }

.bp3-icon-caret-down::before{
  content:"⌄"; }

.bp3-icon-caret-left::before{
  content:"〈"; }

.bp3-icon-caret-right::before{
  content:"〉"; }

.bp3-icon-caret-up::before{
  content:"⌃"; }

.bp3-icon-cell-tower::before{
  content:""; }

.bp3-icon-changes::before{
  content:""; }

.bp3-icon-chart::before{
  content:""; }

.bp3-icon-chat::before{
  content:""; }

.bp3-icon-chevron-backward::before{
  content:""; }

.bp3-icon-chevron-down::before{
  content:""; }

.bp3-icon-chevron-forward::before{
  content:""; }

.bp3-icon-chevron-left::before{
  content:""; }

.bp3-icon-chevron-right::before{
  content:""; }

.bp3-icon-chevron-up::before{
  content:""; }

.bp3-icon-circle::before{
  content:""; }

.bp3-icon-circle-arrow-down::before{
  content:""; }

.bp3-icon-circle-arrow-left::before{
  content:""; }

.bp3-icon-circle-arrow-right::before{
  content:""; }

.bp3-icon-circle-arrow-up::before{
  content:""; }

.bp3-icon-citation::before{
  content:""; }

.bp3-icon-clean::before{
  content:""; }

.bp3-icon-clipboard::before{
  content:""; }

.bp3-icon-cloud::before{
  content:"☁"; }

.bp3-icon-cloud-download::before{
  content:""; }

.bp3-icon-cloud-upload::before{
  content:""; }

.bp3-icon-code::before{
  content:""; }

.bp3-icon-code-block::before{
  content:""; }

.bp3-icon-cog::before{
  content:""; }

.bp3-icon-collapse-all::before{
  content:""; }

.bp3-icon-column-layout::before{
  content:""; }

.bp3-icon-comment::before{
  content:""; }

.bp3-icon-comparison::before{
  content:""; }

.bp3-icon-compass::before{
  content:""; }

.bp3-icon-compressed::before{
  content:""; }

.bp3-icon-confirm::before{
  content:""; }

.bp3-icon-console::before{
  content:""; }

.bp3-icon-contrast::before{
  content:""; }

.bp3-icon-control::before{
  content:""; }

.bp3-icon-credit-card::before{
  content:""; }

.bp3-icon-cross::before{
  content:"✗"; }

.bp3-icon-crown::before{
  content:""; }

.bp3-icon-cube::before{
  content:""; }

.bp3-icon-cube-add::before{
  content:""; }

.bp3-icon-cube-remove::before{
  content:""; }

.bp3-icon-curved-range-chart::before{
  content:""; }

.bp3-icon-cut::before{
  content:""; }

.bp3-icon-dashboard::before{
  content:""; }

.bp3-icon-data-lineage::before{
  content:""; }

.bp3-icon-database::before{
  content:""; }

.bp3-icon-delete::before{
  content:""; }

.bp3-icon-delta::before{
  content:"Δ"; }

.bp3-icon-derive-column::before{
  content:""; }

.bp3-icon-desktop::before{
  content:""; }

.bp3-icon-diagnosis::before{
  content:""; }

.bp3-icon-diagram-tree::before{
  content:""; }

.bp3-icon-direction-left::before{
  content:""; }

.bp3-icon-direction-right::before{
  content:""; }

.bp3-icon-disable::before{
  content:""; }

.bp3-icon-document::before{
  content:""; }

.bp3-icon-document-open::before{
  content:""; }

.bp3-icon-document-share::before{
  content:""; }

.bp3-icon-dollar::before{
  content:"$"; }

.bp3-icon-dot::before{
  content:"•"; }

.bp3-icon-double-caret-horizontal::before{
  content:""; }

.bp3-icon-double-caret-vertical::before{
  content:""; }

.bp3-icon-double-chevron-down::before{
  content:""; }

.bp3-icon-double-chevron-left::before{
  content:""; }

.bp3-icon-double-chevron-right::before{
  content:""; }

.bp3-icon-double-chevron-up::before{
  content:""; }

.bp3-icon-doughnut-chart::before{
  content:""; }

.bp3-icon-download::before{
  content:""; }

.bp3-icon-drag-handle-horizontal::before{
  content:""; }

.bp3-icon-drag-handle-vertical::before{
  content:""; }

.bp3-icon-draw::before{
  content:""; }

.bp3-icon-drive-time::before{
  content:""; }

.bp3-icon-duplicate::before{
  content:""; }

.bp3-icon-edit::before{
  content:"✎"; }

.bp3-icon-eject::before{
  content:"⏏"; }

.bp3-icon-endorsed::before{
  content:""; }

.bp3-icon-envelope::before{
  content:"✉"; }

.bp3-icon-equals::before{
  content:""; }

.bp3-icon-eraser::before{
  content:""; }

.bp3-icon-error::before{
  content:""; }

.bp3-icon-euro::before{
  content:"€"; }

.bp3-icon-exchange::before{
  content:""; }

.bp3-icon-exclude-row::before{
  content:""; }

.bp3-icon-expand-all::before{
  content:""; }

.bp3-icon-export::before{
  content:""; }

.bp3-icon-eye-off::before{
  content:""; }

.bp3-icon-eye-on::before{
  content:""; }

.bp3-icon-eye-open::before{
  content:""; }

.bp3-icon-fast-backward::before{
  content:""; }

.bp3-icon-fast-forward::before{
  content:""; }

.bp3-icon-feed::before{
  content:""; }

.bp3-icon-feed-subscribed::before{
  content:""; }

.bp3-icon-film::before{
  content:""; }

.bp3-icon-filter::before{
  content:""; }

.bp3-icon-filter-keep::before{
  content:""; }

.bp3-icon-filter-list::before{
  content:""; }

.bp3-icon-filter-open::before{
  content:""; }

.bp3-icon-filter-remove::before{
  content:""; }

.bp3-icon-flag::before{
  content:"⚑"; }

.bp3-icon-flame::before{
  content:""; }

.bp3-icon-flash::before{
  content:""; }

.bp3-icon-floppy-disk::before{
  content:""; }

.bp3-icon-flow-branch::before{
  content:""; }

.bp3-icon-flow-end::before{
  content:""; }

.bp3-icon-flow-linear::before{
  content:""; }

.bp3-icon-flow-review::before{
  content:""; }

.bp3-icon-flow-review-branch::before{
  content:""; }

.bp3-icon-flows::before{
  content:""; }

.bp3-icon-folder-close::before{
  content:""; }

.bp3-icon-folder-new::before{
  content:""; }

.bp3-icon-folder-open::before{
  content:""; }

.bp3-icon-folder-shared::before{
  content:""; }

.bp3-icon-folder-shared-open::before{
  content:""; }

.bp3-icon-follower::before{
  content:""; }

.bp3-icon-following::before{
  content:""; }

.bp3-icon-font::before{
  content:""; }

.bp3-icon-fork::before{
  content:""; }

.bp3-icon-form::before{
  content:""; }

.bp3-icon-full-circle::before{
  content:""; }

.bp3-icon-full-stacked-chart::before{
  content:""; }

.bp3-icon-fullscreen::before{
  content:""; }

.bp3-icon-function::before{
  content:""; }

.bp3-icon-gantt-chart::before{
  content:""; }

.bp3-icon-geolocation::before{
  content:""; }

.bp3-icon-geosearch::before{
  content:""; }

.bp3-icon-git-branch::before{
  content:""; }

.bp3-icon-git-commit::before{
  content:""; }

.bp3-icon-git-merge::before{
  content:""; }

.bp3-icon-git-new-branch::before{
  content:""; }

.bp3-icon-git-pull::before{
  content:""; }

.bp3-icon-git-push::before{
  content:""; }

.bp3-icon-git-repo::before{
  content:""; }

.bp3-icon-glass::before{
  content:""; }

.bp3-icon-globe::before{
  content:""; }

.bp3-icon-globe-network::before{
  content:""; }

.bp3-icon-graph::before{
  content:""; }

.bp3-icon-graph-remove::before{
  content:""; }

.bp3-icon-greater-than::before{
  content:""; }

.bp3-icon-greater-than-or-equal-to::before{
  content:""; }

.bp3-icon-grid::before{
  content:""; }

.bp3-icon-grid-view::before{
  content:""; }

.bp3-icon-group-objects::before{
  content:""; }

.bp3-icon-grouped-bar-chart::before{
  content:""; }

.bp3-icon-hand::before{
  content:""; }

.bp3-icon-hand-down::before{
  content:""; }

.bp3-icon-hand-left::before{
  content:""; }

.bp3-icon-hand-right::before{
  content:""; }

.bp3-icon-hand-up::before{
  content:""; }

.bp3-icon-header::before{
  content:""; }

.bp3-icon-header-one::before{
  content:""; }

.bp3-icon-header-two::before{
  content:""; }

.bp3-icon-headset::before{
  content:""; }

.bp3-icon-heart::before{
  content:"♥"; }

.bp3-icon-heart-broken::before{
  content:""; }

.bp3-icon-heat-grid::before{
  content:""; }

.bp3-icon-heatmap::before{
  content:""; }

.bp3-icon-help::before{
  content:"?"; }

.bp3-icon-helper-management::before{
  content:""; }

.bp3-icon-highlight::before{
  content:""; }

.bp3-icon-history::before{
  content:""; }

.bp3-icon-home::before{
  content:"⌂"; }

.bp3-icon-horizontal-bar-chart::before{
  content:""; }

.bp3-icon-horizontal-bar-chart-asc::before{
  content:""; }

.bp3-icon-horizontal-bar-chart-desc::before{
  content:""; }

.bp3-icon-horizontal-distribution::before{
  content:""; }

.bp3-icon-id-number::before{
  content:""; }

.bp3-icon-image-rotate-left::before{
  content:""; }

.bp3-icon-image-rotate-right::before{
  content:""; }

.bp3-icon-import::before{
  content:""; }

.bp3-icon-inbox::before{
  content:""; }

.bp3-icon-inbox-filtered::before{
  content:""; }

.bp3-icon-inbox-geo::before{
  content:""; }

.bp3-icon-inbox-search::before{
  content:""; }

.bp3-icon-inbox-update::before{
  content:""; }

.bp3-icon-info-sign::before{
  content:"ℹ"; }

.bp3-icon-inheritance::before{
  content:""; }

.bp3-icon-inner-join::before{
  content:""; }

.bp3-icon-insert::before{
  content:""; }

.bp3-icon-intersection::before{
  content:""; }

.bp3-icon-ip-address::before{
  content:""; }

.bp3-icon-issue::before{
  content:""; }

.bp3-icon-issue-closed::before{
  content:""; }

.bp3-icon-issue-new::before{
  content:""; }

.bp3-icon-italic::before{
  content:""; }

.bp3-icon-join-table::before{
  content:""; }

.bp3-icon-key::before{
  content:""; }

.bp3-icon-key-backspace::before{
  content:""; }

.bp3-icon-key-command::before{
  content:""; }

.bp3-icon-key-control::before{
  content:""; }

.bp3-icon-key-delete::before{
  content:""; }

.bp3-icon-key-enter::before{
  content:""; }

.bp3-icon-key-escape::before{
  content:""; }

.bp3-icon-key-option::before{
  content:""; }

.bp3-icon-key-shift::before{
  content:""; }

.bp3-icon-key-tab::before{
  content:""; }

.bp3-icon-known-vehicle::before{
  content:""; }

.bp3-icon-lab-test::before{
  content:""; }

.bp3-icon-label::before{
  content:""; }

.bp3-icon-layer::before{
  content:""; }

.bp3-icon-layers::before{
  content:""; }

.bp3-icon-layout::before{
  content:""; }

.bp3-icon-layout-auto::before{
  content:""; }

.bp3-icon-layout-balloon::before{
  content:""; }

.bp3-icon-layout-circle::before{
  content:""; }

.bp3-icon-layout-grid::before{
  content:""; }

.bp3-icon-layout-group-by::before{
  content:""; }

.bp3-icon-layout-hierarchy::before{
  content:""; }

.bp3-icon-layout-linear::before{
  content:""; }

.bp3-icon-layout-skew-grid::before{
  content:""; }

.bp3-icon-layout-sorted-clusters::before{
  content:""; }

.bp3-icon-learning::before{
  content:""; }

.bp3-icon-left-join::before{
  content:""; }

.bp3-icon-less-than::before{
  content:""; }

.bp3-icon-less-than-or-equal-to::before{
  content:""; }

.bp3-icon-lifesaver::before{
  content:""; }

.bp3-icon-lightbulb::before{
  content:""; }

.bp3-icon-link::before{
  content:""; }

.bp3-icon-list::before{
  content:"☰"; }

.bp3-icon-list-columns::before{
  content:""; }

.bp3-icon-list-detail-view::before{
  content:""; }

.bp3-icon-locate::before{
  content:""; }

.bp3-icon-lock::before{
  content:""; }

.bp3-icon-log-in::before{
  content:""; }

.bp3-icon-log-out::before{
  content:""; }

.bp3-icon-manual::before{
  content:""; }

.bp3-icon-manually-entered-data::before{
  content:""; }

.bp3-icon-map::before{
  content:""; }

.bp3-icon-map-create::before{
  content:""; }

.bp3-icon-map-marker::before{
  content:""; }

.bp3-icon-maximize::before{
  content:""; }

.bp3-icon-media::before{
  content:""; }

.bp3-icon-menu::before{
  content:""; }

.bp3-icon-menu-closed::before{
  content:""; }

.bp3-icon-menu-open::before{
  content:""; }

.bp3-icon-merge-columns::before{
  content:""; }

.bp3-icon-merge-links::before{
  content:""; }

.bp3-icon-minimize::before{
  content:""; }

.bp3-icon-minus::before{
  content:"−"; }

.bp3-icon-mobile-phone::before{
  content:""; }

.bp3-icon-mobile-video::before{
  content:""; }

.bp3-icon-moon::before{
  content:""; }

.bp3-icon-more::before{
  content:""; }

.bp3-icon-mountain::before{
  content:""; }

.bp3-icon-move::before{
  content:""; }

.bp3-icon-mugshot::before{
  content:""; }

.bp3-icon-multi-select::before{
  content:""; }

.bp3-icon-music::before{
  content:""; }

.bp3-icon-new-drawing::before{
  content:""; }

.bp3-icon-new-grid-item::before{
  content:""; }

.bp3-icon-new-layer::before{
  content:""; }

.bp3-icon-new-layers::before{
  content:""; }

.bp3-icon-new-link::before{
  content:""; }

.bp3-icon-new-object::before{
  content:""; }

.bp3-icon-new-person::before{
  content:""; }

.bp3-icon-new-prescription::before{
  content:""; }

.bp3-icon-new-text-box::before{
  content:""; }

.bp3-icon-ninja::before{
  content:""; }

.bp3-icon-not-equal-to::before{
  content:""; }

.bp3-icon-notifications::before{
  content:""; }

.bp3-icon-notifications-updated::before{
  content:""; }

.bp3-icon-numbered-list::before{
  content:""; }

.bp3-icon-numerical::before{
  content:""; }

.bp3-icon-office::before{
  content:""; }

.bp3-icon-offline::before{
  content:""; }

.bp3-icon-oil-field::before{
  content:""; }

.bp3-icon-one-column::before{
  content:""; }

.bp3-icon-outdated::before{
  content:""; }

.bp3-icon-page-layout::before{
  content:""; }

.bp3-icon-panel-stats::before{
  content:""; }

.bp3-icon-panel-table::before{
  content:""; }

.bp3-icon-paperclip::before{
  content:""; }

.bp3-icon-paragraph::before{
  content:""; }

.bp3-icon-path::before{
  content:""; }

.bp3-icon-path-search::before{
  content:""; }

.bp3-icon-pause::before{
  content:""; }

.bp3-icon-people::before{
  content:""; }

.bp3-icon-percentage::before{
  content:""; }

.bp3-icon-person::before{
  content:""; }

.bp3-icon-phone::before{
  content:"☎"; }

.bp3-icon-pie-chart::before{
  content:""; }

.bp3-icon-pin::before{
  content:""; }

.bp3-icon-pivot::before{
  content:""; }

.bp3-icon-pivot-table::before{
  content:""; }

.bp3-icon-play::before{
  content:""; }

.bp3-icon-plus::before{
  content:"+"; }

.bp3-icon-polygon-filter::before{
  content:""; }

.bp3-icon-power::before{
  content:""; }

.bp3-icon-predictive-analysis::before{
  content:""; }

.bp3-icon-prescription::before{
  content:""; }

.bp3-icon-presentation::before{
  content:""; }

.bp3-icon-print::before{
  content:"⎙"; }

.bp3-icon-projects::before{
  content:""; }

.bp3-icon-properties::before{
  content:""; }

.bp3-icon-property::before{
  content:""; }

.bp3-icon-publish-function::before{
  content:""; }

.bp3-icon-pulse::before{
  content:""; }

.bp3-icon-random::before{
  content:""; }

.bp3-icon-record::before{
  content:""; }

.bp3-icon-redo::before{
  content:""; }

.bp3-icon-refresh::before{
  content:""; }

.bp3-icon-regression-chart::before{
  content:""; }

.bp3-icon-remove::before{
  content:""; }

.bp3-icon-remove-column::before{
  content:""; }

.bp3-icon-remove-column-left::before{
  content:""; }

.bp3-icon-remove-column-right::before{
  content:""; }

.bp3-icon-remove-row-bottom::before{
  content:""; }

.bp3-icon-remove-row-top::before{
  content:""; }

.bp3-icon-repeat::before{
  content:""; }

.bp3-icon-reset::before{
  content:""; }

.bp3-icon-resolve::before{
  content:""; }

.bp3-icon-rig::before{
  content:""; }

.bp3-icon-right-join::before{
  content:""; }

.bp3-icon-ring::before{
  content:""; }

.bp3-icon-rotate-document::before{
  content:""; }

.bp3-icon-rotate-page::before{
  content:""; }

.bp3-icon-satellite::before{
  content:""; }

.bp3-icon-saved::before{
  content:""; }

.bp3-icon-scatter-plot::before{
  content:""; }

.bp3-icon-search::before{
  content:""; }

.bp3-icon-search-around::before{
  content:""; }

.bp3-icon-search-template::before{
  content:""; }

.bp3-icon-search-text::before{
  content:""; }

.bp3-icon-segmented-control::before{
  content:""; }

.bp3-icon-select::before{
  content:""; }

.bp3-icon-selection::before{
  content:"⦿"; }

.bp3-icon-send-to::before{
  content:""; }

.bp3-icon-send-to-graph::before{
  content:""; }

.bp3-icon-send-to-map::before{
  content:""; }

.bp3-icon-series-add::before{
  content:""; }

.bp3-icon-series-configuration::before{
  content:""; }

.bp3-icon-series-derived::before{
  content:""; }

.bp3-icon-series-filtered::before{
  content:""; }

.bp3-icon-series-search::before{
  content:""; }

.bp3-icon-settings::before{
  content:""; }

.bp3-icon-share::before{
  content:""; }

.bp3-icon-shield::before{
  content:""; }

.bp3-icon-shop::before{
  content:""; }

.bp3-icon-shopping-cart::before{
  content:""; }

.bp3-icon-signal-search::before{
  content:""; }

.bp3-icon-sim-card::before{
  content:""; }

.bp3-icon-slash::before{
  content:""; }

.bp3-icon-small-cross::before{
  content:""; }

.bp3-icon-small-minus::before{
  content:""; }

.bp3-icon-small-plus::before{
  content:""; }

.bp3-icon-small-tick::before{
  content:""; }

.bp3-icon-snowflake::before{
  content:""; }

.bp3-icon-social-media::before{
  content:""; }

.bp3-icon-sort::before{
  content:""; }

.bp3-icon-sort-alphabetical::before{
  content:""; }

.bp3-icon-sort-alphabetical-desc::before{
  content:""; }

.bp3-icon-sort-asc::before{
  content:""; }

.bp3-icon-sort-desc::before{
  content:""; }

.bp3-icon-sort-numerical::before{
  content:""; }

.bp3-icon-sort-numerical-desc::before{
  content:""; }

.bp3-icon-split-columns::before{
  content:""; }

.bp3-icon-square::before{
  content:""; }

.bp3-icon-stacked-chart::before{
  content:""; }

.bp3-icon-star::before{
  content:"★"; }

.bp3-icon-star-empty::before{
  content:"☆"; }

.bp3-icon-step-backward::before{
  content:""; }

.bp3-icon-step-chart::before{
  content:""; }

.bp3-icon-step-forward::before{
  content:""; }

.bp3-icon-stop::before{
  content:""; }

.bp3-icon-stopwatch::before{
  content:""; }

.bp3-icon-strikethrough::before{
  content:""; }

.bp3-icon-style::before{
  content:""; }

.bp3-icon-swap-horizontal::before{
  content:""; }

.bp3-icon-swap-vertical::before{
  content:""; }

.bp3-icon-symbol-circle::before{
  content:""; }

.bp3-icon-symbol-cross::before{
  content:""; }

.bp3-icon-symbol-diamond::before{
  content:""; }

.bp3-icon-symbol-square::before{
  content:""; }

.bp3-icon-symbol-triangle-down::before{
  content:""; }

.bp3-icon-symbol-triangle-up::before{
  content:""; }

.bp3-icon-tag::before{
  content:""; }

.bp3-icon-take-action::before{
  content:""; }

.bp3-icon-taxi::before{
  content:""; }

.bp3-icon-text-highlight::before{
  content:""; }

.bp3-icon-th::before{
  content:""; }

.bp3-icon-th-derived::before{
  content:""; }

.bp3-icon-th-disconnect::before{
  content:""; }

.bp3-icon-th-filtered::before{
  content:""; }

.bp3-icon-th-list::before{
  content:""; }

.bp3-icon-thumbs-down::before{
  content:""; }

.bp3-icon-thumbs-up::before{
  content:""; }

.bp3-icon-tick::before{
  content:"✓"; }

.bp3-icon-tick-circle::before{
  content:""; }

.bp3-icon-time::before{
  content:"⏲"; }

.bp3-icon-timeline-area-chart::before{
  content:""; }

.bp3-icon-timeline-bar-chart::before{
  content:""; }

.bp3-icon-timeline-events::before{
  content:""; }

.bp3-icon-timeline-line-chart::before{
  content:""; }

.bp3-icon-tint::before{
  content:""; }

.bp3-icon-torch::before{
  content:""; }

.bp3-icon-tractor::before{
  content:""; }

.bp3-icon-train::before{
  content:""; }

.bp3-icon-translate::before{
  content:""; }

.bp3-icon-trash::before{
  content:""; }

.bp3-icon-tree::before{
  content:""; }

.bp3-icon-trending-down::before{
  content:""; }

.bp3-icon-trending-up::before{
  content:""; }

.bp3-icon-truck::before{
  content:""; }

.bp3-icon-two-columns::before{
  content:""; }

.bp3-icon-unarchive::before{
  content:""; }

.bp3-icon-underline::before{
  content:"⎁"; }

.bp3-icon-undo::before{
  content:"⎌"; }

.bp3-icon-ungroup-objects::before{
  content:""; }

.bp3-icon-unknown-vehicle::before{
  content:""; }

.bp3-icon-unlock::before{
  content:""; }

.bp3-icon-unpin::before{
  content:""; }

.bp3-icon-unresolve::before{
  content:""; }

.bp3-icon-updated::before{
  content:""; }

.bp3-icon-upload::before{
  content:""; }

.bp3-icon-user::before{
  content:""; }

.bp3-icon-variable::before{
  content:""; }

.bp3-icon-vertical-bar-chart-asc::before{
  content:""; }

.bp3-icon-vertical-bar-chart-desc::before{
  content:""; }

.bp3-icon-vertical-distribution::before{
  content:""; }

.bp3-icon-video::before{
  content:""; }

.bp3-icon-volume-down::before{
  content:""; }

.bp3-icon-volume-off::before{
  content:""; }

.bp3-icon-volume-up::before{
  content:""; }

.bp3-icon-walk::before{
  content:""; }

.bp3-icon-warning-sign::before{
  content:""; }

.bp3-icon-waterfall-chart::before{
  content:""; }

.bp3-icon-widget::before{
  content:""; }

.bp3-icon-widget-button::before{
  content:""; }

.bp3-icon-widget-footer::before{
  content:""; }

.bp3-icon-widget-header::before{
  content:""; }

.bp3-icon-wrench::before{
  content:""; }

.bp3-icon-zoom-in::before{
  content:""; }

.bp3-icon-zoom-out::before{
  content:""; }

.bp3-icon-zoom-to-fit::before{
  content:""; }
.bp3-submenu > .bp3-popover-wrapper{
  display:block; }

.bp3-submenu .bp3-popover-target{
  display:block; }
  .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-menu-item{ }

.bp3-submenu.bp3-popover{
  -webkit-box-shadow:none;
          box-shadow:none;
  padding:0 5px; }
  .bp3-submenu.bp3-popover > .bp3-popover-content{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 2px 4px rgba(16, 22, 26, 0.2), 0 8px 24px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 2px 4px rgba(16, 22, 26, 0.2), 0 8px 24px rgba(16, 22, 26, 0.2); }
  .bp3-dark .bp3-submenu.bp3-popover, .bp3-submenu.bp3-popover.bp3-dark{
    -webkit-box-shadow:none;
            box-shadow:none; }
    .bp3-dark .bp3-submenu.bp3-popover > .bp3-popover-content, .bp3-submenu.bp3-popover.bp3-dark > .bp3-popover-content{
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 2px 4px rgba(16, 22, 26, 0.4), 0 8px 24px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 2px 4px rgba(16, 22, 26, 0.4), 0 8px 24px rgba(16, 22, 26, 0.4); }
.bp3-menu{
  background:#ffffff;
  border-radius:3px;
  color:#182026;
  list-style:none;
  margin:0;
  min-width:180px;
  padding:5px;
  text-align:left; }

.bp3-menu-divider{
  border-top:1px solid rgba(16, 22, 26, 0.15);
  display:block;
  margin:5px; }
  .bp3-dark .bp3-menu-divider{
    border-top-color:rgba(255, 255, 255, 0.15); }

.bp3-menu-item{
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-orient:horizontal;
  -webkit-box-direction:normal;
      -ms-flex-direction:row;
          flex-direction:row;
  -webkit-box-align:start;
      -ms-flex-align:start;
          align-items:flex-start;
  border-radius:2px;
  color:inherit;
  line-height:20px;
  padding:5px 7px;
  text-decoration:none;
  -webkit-user-select:none;
     -moz-user-select:none;
      -ms-user-select:none;
          user-select:none; }
  .bp3-menu-item > *{
    -webkit-box-flex:0;
        -ms-flex-positive:0;
            flex-grow:0;
    -ms-flex-negative:0;
        flex-shrink:0; }
  .bp3-menu-item > .bp3-fill{
    -webkit-box-flex:1;
        -ms-flex-positive:1;
            flex-grow:1;
    -ms-flex-negative:1;
        flex-shrink:1; }
  .bp3-menu-item::before,
  .bp3-menu-item > *{
    margin-right:7px; }
  .bp3-menu-item:empty::before,
  .bp3-menu-item > :last-child{
    margin-right:0; }
  .bp3-menu-item > .bp3-fill{
    word-break:break-word; }
  .bp3-menu-item:hover, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-menu-item{
    background-color:rgba(167, 182, 194, 0.3);
    cursor:pointer;
    text-decoration:none; }
  .bp3-menu-item.bp3-disabled{
    background-color:inherit;
    color:rgba(92, 112, 128, 0.6);
    cursor:not-allowed; }
  .bp3-dark .bp3-menu-item{
    color:inherit; }
    .bp3-dark .bp3-menu-item:hover, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-menu-item, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-menu-item{
      background-color:rgba(138, 155, 168, 0.15);
      color:inherit; }
    .bp3-dark .bp3-menu-item.bp3-disabled{
      background-color:inherit;
      color:rgba(167, 182, 194, 0.6); }
  .bp3-menu-item.bp3-intent-primary{
    color:#106ba3; }
    .bp3-menu-item.bp3-intent-primary .bp3-icon{
      color:inherit; }
    .bp3-menu-item.bp3-intent-primary::before, .bp3-menu-item.bp3-intent-primary::after,
    .bp3-menu-item.bp3-intent-primary .bp3-menu-item-label{
      color:#106ba3; }
    .bp3-menu-item.bp3-intent-primary:hover, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item, .bp3-menu-item.bp3-intent-primary.bp3-active{
      background-color:#137cbd; }
    .bp3-menu-item.bp3-intent-primary:active{
      background-color:#106ba3; }
    .bp3-menu-item.bp3-intent-primary:hover, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item, .bp3-menu-item.bp3-intent-primary:hover::before, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item::before, .bp3-menu-item.bp3-intent-primary:hover::after, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item::after,
    .bp3-menu-item.bp3-intent-primary:hover .bp3-menu-item-label,
    .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item .bp3-menu-item-label, .bp3-menu-item.bp3-intent-primary:active, .bp3-menu-item.bp3-intent-primary:active::before, .bp3-menu-item.bp3-intent-primary:active::after,
    .bp3-menu-item.bp3-intent-primary:active .bp3-menu-item-label, .bp3-menu-item.bp3-intent-primary.bp3-active, .bp3-menu-item.bp3-intent-primary.bp3-active::before, .bp3-menu-item.bp3-intent-primary.bp3-active::after,
    .bp3-menu-item.bp3-intent-primary.bp3-active .bp3-menu-item-label{
      color:#ffffff; }
  .bp3-menu-item.bp3-intent-success{
    color:#0d8050; }
    .bp3-menu-item.bp3-intent-success .bp3-icon{
      color:inherit; }
    .bp3-menu-item.bp3-intent-success::before, .bp3-menu-item.bp3-intent-success::after,
    .bp3-menu-item.bp3-intent-success .bp3-menu-item-label{
      color:#0d8050; }
    .bp3-menu-item.bp3-intent-success:hover, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item, .bp3-menu-item.bp3-intent-success.bp3-active{
      background-color:#0f9960; }
    .bp3-menu-item.bp3-intent-success:active{
      background-color:#0d8050; }
    .bp3-menu-item.bp3-intent-success:hover, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item, .bp3-menu-item.bp3-intent-success:hover::before, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item::before, .bp3-menu-item.bp3-intent-success:hover::after, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item::after,
    .bp3-menu-item.bp3-intent-success:hover .bp3-menu-item-label,
    .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item .bp3-menu-item-label, .bp3-menu-item.bp3-intent-success:active, .bp3-menu-item.bp3-intent-success:active::before, .bp3-menu-item.bp3-intent-success:active::after,
    .bp3-menu-item.bp3-intent-success:active .bp3-menu-item-label, .bp3-menu-item.bp3-intent-success.bp3-active, .bp3-menu-item.bp3-intent-success.bp3-active::before, .bp3-menu-item.bp3-intent-success.bp3-active::after,
    .bp3-menu-item.bp3-intent-success.bp3-active .bp3-menu-item-label{
      color:#ffffff; }
  .bp3-menu-item.bp3-intent-warning{
    color:#bf7326; }
    .bp3-menu-item.bp3-intent-warning .bp3-icon{
      color:inherit; }
    .bp3-menu-item.bp3-intent-warning::before, .bp3-menu-item.bp3-intent-warning::after,
    .bp3-menu-item.bp3-intent-warning .bp3-menu-item-label{
      color:#bf7326; }
    .bp3-menu-item.bp3-intent-warning:hover, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item, .bp3-menu-item.bp3-intent-warning.bp3-active{
      background-color:#d9822b; }
    .bp3-menu-item.bp3-intent-warning:active{
      background-color:#bf7326; }
    .bp3-menu-item.bp3-intent-warning:hover, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item, .bp3-menu-item.bp3-intent-warning:hover::before, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item::before, .bp3-menu-item.bp3-intent-warning:hover::after, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item::after,
    .bp3-menu-item.bp3-intent-warning:hover .bp3-menu-item-label,
    .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item .bp3-menu-item-label, .bp3-menu-item.bp3-intent-warning:active, .bp3-menu-item.bp3-intent-warning:active::before, .bp3-menu-item.bp3-intent-warning:active::after,
    .bp3-menu-item.bp3-intent-warning:active .bp3-menu-item-label, .bp3-menu-item.bp3-intent-warning.bp3-active, .bp3-menu-item.bp3-intent-warning.bp3-active::before, .bp3-menu-item.bp3-intent-warning.bp3-active::after,
    .bp3-menu-item.bp3-intent-warning.bp3-active .bp3-menu-item-label{
      color:#ffffff; }
  .bp3-menu-item.bp3-intent-danger{
    color:#c23030; }
    .bp3-menu-item.bp3-intent-danger .bp3-icon{
      color:inherit; }
    .bp3-menu-item.bp3-intent-danger::before, .bp3-menu-item.bp3-intent-danger::after,
    .bp3-menu-item.bp3-intent-danger .bp3-menu-item-label{
      color:#c23030; }
    .bp3-menu-item.bp3-intent-danger:hover, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item, .bp3-menu-item.bp3-intent-danger.bp3-active{
      background-color:#db3737; }
    .bp3-menu-item.bp3-intent-danger:active{
      background-color:#c23030; }
    .bp3-menu-item.bp3-intent-danger:hover, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item, .bp3-menu-item.bp3-intent-danger:hover::before, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item::before, .bp3-menu-item.bp3-intent-danger:hover::after, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item::after,
    .bp3-menu-item.bp3-intent-danger:hover .bp3-menu-item-label,
    .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item .bp3-menu-item-label, .bp3-menu-item.bp3-intent-danger:active, .bp3-menu-item.bp3-intent-danger:active::before, .bp3-menu-item.bp3-intent-danger:active::after,
    .bp3-menu-item.bp3-intent-danger:active .bp3-menu-item-label, .bp3-menu-item.bp3-intent-danger.bp3-active, .bp3-menu-item.bp3-intent-danger.bp3-active::before, .bp3-menu-item.bp3-intent-danger.bp3-active::after,
    .bp3-menu-item.bp3-intent-danger.bp3-active .bp3-menu-item-label{
      color:#ffffff; }
  .bp3-menu-item::before{
    font-family:"Icons16", sans-serif;
    font-size:16px;
    font-style:normal;
    font-weight:400;
    line-height:1;
    -moz-osx-font-smoothing:grayscale;
    -webkit-font-smoothing:antialiased;
    margin-right:7px; }
  .bp3-menu-item::before,
  .bp3-menu-item > .bp3-icon{
    color:#5c7080;
    margin-top:2px; }
  .bp3-menu-item .bp3-menu-item-label{
    color:#5c7080; }
  .bp3-menu-item:hover, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-menu-item{
    color:inherit; }
  .bp3-menu-item.bp3-active, .bp3-menu-item:active{
    background-color:rgba(115, 134, 148, 0.3); }
  .bp3-menu-item.bp3-disabled{
    background-color:inherit !important;
    color:rgba(92, 112, 128, 0.6) !important;
    cursor:not-allowed !important;
    outline:none !important; }
    .bp3-menu-item.bp3-disabled::before,
    .bp3-menu-item.bp3-disabled > .bp3-icon,
    .bp3-menu-item.bp3-disabled .bp3-menu-item-label{
      color:rgba(92, 112, 128, 0.6) !important; }
  .bp3-large .bp3-menu-item{
    font-size:16px;
    line-height:22px;
    padding:9px 7px; }
    .bp3-large .bp3-menu-item .bp3-icon{
      margin-top:3px; }
    .bp3-large .bp3-menu-item::before{
      font-family:"Icons20", sans-serif;
      font-size:20px;
      font-style:normal;
      font-weight:400;
      line-height:1;
      -moz-osx-font-smoothing:grayscale;
      -webkit-font-smoothing:antialiased;
      margin-right:10px;
      margin-top:1px; }

button.bp3-menu-item{
  background:none;
  border:none;
  text-align:left;
  width:100%; }
.bp3-menu-header{
  border-top:1px solid rgba(16, 22, 26, 0.15);
  display:block;
  margin:5px;
  cursor:default;
  padding-left:2px; }
  .bp3-dark .bp3-menu-header{
    border-top-color:rgba(255, 255, 255, 0.15); }
  .bp3-menu-header:first-of-type{
    border-top:none; }
  .bp3-menu-header > h6{
    color:#182026;
    font-weight:600;
    overflow:hidden;
    text-overflow:ellipsis;
    white-space:nowrap;
    word-wrap:normal;
    line-height:17px;
    margin:0;
    padding:10px 7px 0 1px; }
    .bp3-dark .bp3-menu-header > h6{
      color:#f5f8fa; }
  .bp3-menu-header:first-of-type > h6{
    padding-top:0; }
  .bp3-large .bp3-menu-header > h6{
    font-size:18px;
    padding-bottom:5px;
    padding-top:15px; }
  .bp3-large .bp3-menu-header:first-of-type > h6{
    padding-top:0; }

.bp3-dark .bp3-menu{
  background:#30404d;
  color:#f5f8fa; }

.bp3-dark .bp3-menu-item{ }
  .bp3-dark .bp3-menu-item.bp3-intent-primary{
    color:#48aff0; }
    .bp3-dark .bp3-menu-item.bp3-intent-primary .bp3-icon{
      color:inherit; }
    .bp3-dark .bp3-menu-item.bp3-intent-primary::before, .bp3-dark .bp3-menu-item.bp3-intent-primary::after,
    .bp3-dark .bp3-menu-item.bp3-intent-primary .bp3-menu-item-label{
      color:#48aff0; }
    .bp3-dark .bp3-menu-item.bp3-intent-primary:hover, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item, .bp3-dark .bp3-menu-item.bp3-intent-primary.bp3-active{
      background-color:#137cbd; }
    .bp3-dark .bp3-menu-item.bp3-intent-primary:active{
      background-color:#106ba3; }
    .bp3-dark .bp3-menu-item.bp3-intent-primary:hover, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item, .bp3-dark .bp3-menu-item.bp3-intent-primary:hover::before, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item::before, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item::before, .bp3-dark .bp3-menu-item.bp3-intent-primary:hover::after, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item::after, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item::after,
    .bp3-dark .bp3-menu-item.bp3-intent-primary:hover .bp3-menu-item-label,
    .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item .bp3-menu-item-label,
    .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item .bp3-menu-item-label, .bp3-dark .bp3-menu-item.bp3-intent-primary:active, .bp3-dark .bp3-menu-item.bp3-intent-primary:active::before, .bp3-dark .bp3-menu-item.bp3-intent-primary:active::after,
    .bp3-dark .bp3-menu-item.bp3-intent-primary:active .bp3-menu-item-label, .bp3-dark .bp3-menu-item.bp3-intent-primary.bp3-active, .bp3-dark .bp3-menu-item.bp3-intent-primary.bp3-active::before, .bp3-dark .bp3-menu-item.bp3-intent-primary.bp3-active::after,
    .bp3-dark .bp3-menu-item.bp3-intent-primary.bp3-active .bp3-menu-item-label{
      color:#ffffff; }
  .bp3-dark .bp3-menu-item.bp3-intent-success{
    color:#3dcc91; }
    .bp3-dark .bp3-menu-item.bp3-intent-success .bp3-icon{
      color:inherit; }
    .bp3-dark .bp3-menu-item.bp3-intent-success::before, .bp3-dark .bp3-menu-item.bp3-intent-success::after,
    .bp3-dark .bp3-menu-item.bp3-intent-success .bp3-menu-item-label{
      color:#3dcc91; }
    .bp3-dark .bp3-menu-item.bp3-intent-success:hover, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item, .bp3-dark .bp3-menu-item.bp3-intent-success.bp3-active{
      background-color:#0f9960; }
    .bp3-dark .bp3-menu-item.bp3-intent-success:active{
      background-color:#0d8050; }
    .bp3-dark .bp3-menu-item.bp3-intent-success:hover, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item, .bp3-dark .bp3-menu-item.bp3-intent-success:hover::before, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item::before, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item::before, .bp3-dark .bp3-menu-item.bp3-intent-success:hover::after, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item::after, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item::after,
    .bp3-dark .bp3-menu-item.bp3-intent-success:hover .bp3-menu-item-label,
    .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item .bp3-menu-item-label,
    .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item .bp3-menu-item-label, .bp3-dark .bp3-menu-item.bp3-intent-success:active, .bp3-dark .bp3-menu-item.bp3-intent-success:active::before, .bp3-dark .bp3-menu-item.bp3-intent-success:active::after,
    .bp3-dark .bp3-menu-item.bp3-intent-success:active .bp3-menu-item-label, .bp3-dark .bp3-menu-item.bp3-intent-success.bp3-active, .bp3-dark .bp3-menu-item.bp3-intent-success.bp3-active::before, .bp3-dark .bp3-menu-item.bp3-intent-success.bp3-active::after,
    .bp3-dark .bp3-menu-item.bp3-intent-success.bp3-active .bp3-menu-item-label{
      color:#ffffff; }
  .bp3-dark .bp3-menu-item.bp3-intent-warning{
    color:#ffb366; }
    .bp3-dark .bp3-menu-item.bp3-intent-warning .bp3-icon{
      color:inherit; }
    .bp3-dark .bp3-menu-item.bp3-intent-warning::before, .bp3-dark .bp3-menu-item.bp3-intent-warning::after,
    .bp3-dark .bp3-menu-item.bp3-intent-warning .bp3-menu-item-label{
      color:#ffb366; }
    .bp3-dark .bp3-menu-item.bp3-intent-warning:hover, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item, .bp3-dark .bp3-menu-item.bp3-intent-warning.bp3-active{
      background-color:#d9822b; }
    .bp3-dark .bp3-menu-item.bp3-intent-warning:active{
      background-color:#bf7326; }
    .bp3-dark .bp3-menu-item.bp3-intent-warning:hover, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item, .bp3-dark .bp3-menu-item.bp3-intent-warning:hover::before, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item::before, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item::before, .bp3-dark .bp3-menu-item.bp3-intent-warning:hover::after, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item::after, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item::after,
    .bp3-dark .bp3-menu-item.bp3-intent-warning:hover .bp3-menu-item-label,
    .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item .bp3-menu-item-label,
    .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item .bp3-menu-item-label, .bp3-dark .bp3-menu-item.bp3-intent-warning:active, .bp3-dark .bp3-menu-item.bp3-intent-warning:active::before, .bp3-dark .bp3-menu-item.bp3-intent-warning:active::after,
    .bp3-dark .bp3-menu-item.bp3-intent-warning:active .bp3-menu-item-label, .bp3-dark .bp3-menu-item.bp3-intent-warning.bp3-active, .bp3-dark .bp3-menu-item.bp3-intent-warning.bp3-active::before, .bp3-dark .bp3-menu-item.bp3-intent-warning.bp3-active::after,
    .bp3-dark .bp3-menu-item.bp3-intent-warning.bp3-active .bp3-menu-item-label{
      color:#ffffff; }
  .bp3-dark .bp3-menu-item.bp3-intent-danger{
    color:#ff7373; }
    .bp3-dark .bp3-menu-item.bp3-intent-danger .bp3-icon{
      color:inherit; }
    .bp3-dark .bp3-menu-item.bp3-intent-danger::before, .bp3-dark .bp3-menu-item.bp3-intent-danger::after,
    .bp3-dark .bp3-menu-item.bp3-intent-danger .bp3-menu-item-label{
      color:#ff7373; }
    .bp3-dark .bp3-menu-item.bp3-intent-danger:hover, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item, .bp3-dark .bp3-menu-item.bp3-intent-danger.bp3-active{
      background-color:#db3737; }
    .bp3-dark .bp3-menu-item.bp3-intent-danger:active{
      background-color:#c23030; }
    .bp3-dark .bp3-menu-item.bp3-intent-danger:hover, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item, .bp3-dark .bp3-menu-item.bp3-intent-danger:hover::before, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item::before, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item::before, .bp3-dark .bp3-menu-item.bp3-intent-danger:hover::after, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item::after, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item::after,
    .bp3-dark .bp3-menu-item.bp3-intent-danger:hover .bp3-menu-item-label,
    .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item .bp3-menu-item-label,
    .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item .bp3-menu-item-label, .bp3-dark .bp3-menu-item.bp3-intent-danger:active, .bp3-dark .bp3-menu-item.bp3-intent-danger:active::before, .bp3-dark .bp3-menu-item.bp3-intent-danger:active::after,
    .bp3-dark .bp3-menu-item.bp3-intent-danger:active .bp3-menu-item-label, .bp3-dark .bp3-menu-item.bp3-intent-danger.bp3-active, .bp3-dark .bp3-menu-item.bp3-intent-danger.bp3-active::before, .bp3-dark .bp3-menu-item.bp3-intent-danger.bp3-active::after,
    .bp3-dark .bp3-menu-item.bp3-intent-danger.bp3-active .bp3-menu-item-label{
      color:#ffffff; }
  .bp3-dark .bp3-menu-item::before,
  .bp3-dark .bp3-menu-item > .bp3-icon{
    color:#a7b6c2; }
  .bp3-dark .bp3-menu-item .bp3-menu-item-label{
    color:#a7b6c2; }
  .bp3-dark .bp3-menu-item.bp3-active, .bp3-dark .bp3-menu-item:active{
    background-color:rgba(138, 155, 168, 0.3); }
  .bp3-dark .bp3-menu-item.bp3-disabled{
    color:rgba(167, 182, 194, 0.6) !important; }
    .bp3-dark .bp3-menu-item.bp3-disabled::before,
    .bp3-dark .bp3-menu-item.bp3-disabled > .bp3-icon,
    .bp3-dark .bp3-menu-item.bp3-disabled .bp3-menu-item-label{
      color:rgba(167, 182, 194, 0.6) !important; }

.bp3-dark .bp3-menu-divider,
.bp3-dark .bp3-menu-header{
  border-color:rgba(255, 255, 255, 0.15); }

.bp3-dark .bp3-menu-header > h6{
  color:#f5f8fa; }

.bp3-label .bp3-menu{
  margin-top:5px; }
.bp3-navbar{
  background-color:#ffffff;
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.2);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.2);
  height:50px;
  padding:0 15px;
  position:relative;
  width:100%;
  z-index:10; }
  .bp3-navbar.bp3-dark,
  .bp3-dark .bp3-navbar{
    background-color:#394b59; }
  .bp3-navbar.bp3-dark{
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.4);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.4); }
  .bp3-dark .bp3-navbar{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.4); }
  .bp3-navbar.bp3-fixed-top{
    left:0;
    position:fixed;
    right:0;
    top:0; }

.bp3-navbar-heading{
  font-size:16px;
  margin-right:15px; }

.bp3-navbar-group{
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  height:50px; }
  .bp3-navbar-group.bp3-align-left{
    float:left; }
  .bp3-navbar-group.bp3-align-right{
    float:right; }

.bp3-navbar-divider{
  border-left:1px solid rgba(16, 22, 26, 0.15);
  height:20px;
  margin:0 10px; }
  .bp3-dark .bp3-navbar-divider{
    border-left-color:rgba(255, 255, 255, 0.15); }
.bp3-non-ideal-state{
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-orient:vertical;
  -webkit-box-direction:normal;
      -ms-flex-direction:column;
          flex-direction:column;
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  height:100%;
  -webkit-box-pack:center;
      -ms-flex-pack:center;
          justify-content:center;
  text-align:center;
  width:100%; }
  .bp3-non-ideal-state > *{
    -webkit-box-flex:0;
        -ms-flex-positive:0;
            flex-grow:0;
    -ms-flex-negative:0;
        flex-shrink:0; }
  .bp3-non-ideal-state > .bp3-fill{
    -webkit-box-flex:1;
        -ms-flex-positive:1;
            flex-grow:1;
    -ms-flex-negative:1;
        flex-shrink:1; }
  .bp3-non-ideal-state::before,
  .bp3-non-ideal-state > *{
    margin-bottom:20px; }
  .bp3-non-ideal-state:empty::before,
  .bp3-non-ideal-state > :last-child{
    margin-bottom:0; }
  .bp3-non-ideal-state > *{
    max-width:400px; }

.bp3-non-ideal-state-visual{
  color:rgba(92, 112, 128, 0.6);
  font-size:60px; }
  .bp3-dark .bp3-non-ideal-state-visual{
    color:rgba(167, 182, 194, 0.6); }

.bp3-overflow-list{
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -ms-flex-wrap:nowrap;
      flex-wrap:nowrap;
  min-width:0; }

.bp3-overflow-list-spacer{
  -ms-flex-negative:1;
      flex-shrink:1;
  width:1px; }

body.bp3-overlay-open{
  overflow:hidden; }

.bp3-overlay{
  bottom:0;
  left:0;
  position:static;
  right:0;
  top:0;
  z-index:20; }
  .bp3-overlay:not(.bp3-overlay-open){
    pointer-events:none; }
  .bp3-overlay.bp3-overlay-container{
    overflow:hidden;
    position:fixed; }
    .bp3-overlay.bp3-overlay-container.bp3-overlay-inline{
      position:absolute; }
  .bp3-overlay.bp3-overlay-scroll-container{
    overflow:auto;
    position:fixed; }
    .bp3-overlay.bp3-overlay-scroll-container.bp3-overlay-inline{
      position:absolute; }
  .bp3-overlay.bp3-overlay-inline{
    display:inline;
    overflow:visible; }

.bp3-overlay-content{
  position:fixed;
  z-index:20; }
  .bp3-overlay-inline .bp3-overlay-content,
  .bp3-overlay-scroll-container .bp3-overlay-content{
    position:absolute; }

.bp3-overlay-backdrop{
  bottom:0;
  left:0;
  position:fixed;
  right:0;
  top:0;
  opacity:1;
  background-color:rgba(16, 22, 26, 0.7);
  overflow:auto;
  -webkit-user-select:none;
     -moz-user-select:none;
      -ms-user-select:none;
          user-select:none;
  z-index:20; }
  .bp3-overlay-backdrop.bp3-overlay-enter, .bp3-overlay-backdrop.bp3-overlay-appear{
    opacity:0; }
  .bp3-overlay-backdrop.bp3-overlay-enter-active, .bp3-overlay-backdrop.bp3-overlay-appear-active{
    opacity:1;
    -webkit-transition-delay:0;
            transition-delay:0;
    -webkit-transition-duration:200ms;
            transition-duration:200ms;
    -webkit-transition-property:opacity;
    transition-property:opacity;
    -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
            transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-overlay-backdrop.bp3-overlay-exit{
    opacity:1; }
  .bp3-overlay-backdrop.bp3-overlay-exit-active{
    opacity:0;
    -webkit-transition-delay:0;
            transition-delay:0;
    -webkit-transition-duration:200ms;
            transition-duration:200ms;
    -webkit-transition-property:opacity;
    transition-property:opacity;
    -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
            transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-overlay-backdrop:focus{
    outline:none; }
  .bp3-overlay-inline .bp3-overlay-backdrop{
    position:absolute; }
.bp3-panel-stack{
  overflow:hidden;
  position:relative; }

.bp3-panel-stack-header{
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  -webkit-box-shadow:0 1px rgba(16, 22, 26, 0.15);
          box-shadow:0 1px rgba(16, 22, 26, 0.15);
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -ms-flex-negative:0;
      flex-shrink:0;
  height:30px;
  z-index:1; }
  .bp3-dark .bp3-panel-stack-header{
    -webkit-box-shadow:0 1px rgba(255, 255, 255, 0.15);
            box-shadow:0 1px rgba(255, 255, 255, 0.15); }
  .bp3-panel-stack-header > span{
    -webkit-box-align:stretch;
        -ms-flex-align:stretch;
            align-items:stretch;
    display:-webkit-box;
    display:-ms-flexbox;
    display:flex;
    -webkit-box-flex:1;
        -ms-flex:1;
            flex:1; }
  .bp3-panel-stack-header .bp3-heading{
    margin:0 5px; }

.bp3-button.bp3-panel-stack-header-back{
  margin-left:5px;
  padding-left:0;
  white-space:nowrap; }
  .bp3-button.bp3-panel-stack-header-back .bp3-icon{
    margin:0 2px; }

.bp3-panel-stack-view{
  bottom:0;
  left:0;
  position:absolute;
  right:0;
  top:0;
  background-color:#ffffff;
  border-right:1px solid rgba(16, 22, 26, 0.15);
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-orient:vertical;
  -webkit-box-direction:normal;
      -ms-flex-direction:column;
          flex-direction:column;
  margin-right:-1px;
  overflow-y:auto;
  z-index:1; }
  .bp3-dark .bp3-panel-stack-view{
    background-color:#30404d; }
  .bp3-panel-stack-view:nth-last-child(n + 4){
    display:none; }

.bp3-panel-stack-push .bp3-panel-stack-enter, .bp3-panel-stack-push .bp3-panel-stack-appear{
  -webkit-transform:translateX(100%);
          transform:translateX(100%);
  opacity:0; }

.bp3-panel-stack-push .bp3-panel-stack-enter-active, .bp3-panel-stack-push .bp3-panel-stack-appear-active{
  -webkit-transform:translate(0%);
          transform:translate(0%);
  opacity:1;
  -webkit-transition-delay:0;
          transition-delay:0;
  -webkit-transition-duration:400ms;
          transition-duration:400ms;
  -webkit-transition-property:opacity, -webkit-transform;
  transition-property:opacity, -webkit-transform;
  transition-property:transform, opacity;
  transition-property:transform, opacity, -webkit-transform;
  -webkit-transition-timing-function:ease;
          transition-timing-function:ease; }

.bp3-panel-stack-push .bp3-panel-stack-exit{
  -webkit-transform:translate(0%);
          transform:translate(0%);
  opacity:1; }

.bp3-panel-stack-push .bp3-panel-stack-exit-active{
  -webkit-transform:translateX(-50%);
          transform:translateX(-50%);
  opacity:0;
  -webkit-transition-delay:0;
          transition-delay:0;
  -webkit-transition-duration:400ms;
          transition-duration:400ms;
  -webkit-transition-property:opacity, -webkit-transform;
  transition-property:opacity, -webkit-transform;
  transition-property:transform, opacity;
  transition-property:transform, opacity, -webkit-transform;
  -webkit-transition-timing-function:ease;
          transition-timing-function:ease; }

.bp3-panel-stack-pop .bp3-panel-stack-enter, .bp3-panel-stack-pop .bp3-panel-stack-appear{
  -webkit-transform:translateX(-50%);
          transform:translateX(-50%);
  opacity:0; }

.bp3-panel-stack-pop .bp3-panel-stack-enter-active, .bp3-panel-stack-pop .bp3-panel-stack-appear-active{
  -webkit-transform:translate(0%);
          transform:translate(0%);
  opacity:1;
  -webkit-transition-delay:0;
          transition-delay:0;
  -webkit-transition-duration:400ms;
          transition-duration:400ms;
  -webkit-transition-property:opacity, -webkit-transform;
  transition-property:opacity, -webkit-transform;
  transition-property:transform, opacity;
  transition-property:transform, opacity, -webkit-transform;
  -webkit-transition-timing-function:ease;
          transition-timing-function:ease; }

.bp3-panel-stack-pop .bp3-panel-stack-exit{
  -webkit-transform:translate(0%);
          transform:translate(0%);
  opacity:1; }

.bp3-panel-stack-pop .bp3-panel-stack-exit-active{
  -webkit-transform:translateX(100%);
          transform:translateX(100%);
  opacity:0;
  -webkit-transition-delay:0;
          transition-delay:0;
  -webkit-transition-duration:400ms;
          transition-duration:400ms;
  -webkit-transition-property:opacity, -webkit-transform;
  transition-property:opacity, -webkit-transform;
  transition-property:transform, opacity;
  transition-property:transform, opacity, -webkit-transform;
  -webkit-transition-timing-function:ease;
          transition-timing-function:ease; }
.bp3-panel-stack2{
  overflow:hidden;
  position:relative; }

.bp3-panel-stack2-header{
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  -webkit-box-shadow:0 1px rgba(16, 22, 26, 0.15);
          box-shadow:0 1px rgba(16, 22, 26, 0.15);
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -ms-flex-negative:0;
      flex-shrink:0;
  height:30px;
  z-index:1; }
  .bp3-dark .bp3-panel-stack2-header{
    -webkit-box-shadow:0 1px rgba(255, 255, 255, 0.15);
            box-shadow:0 1px rgba(255, 255, 255, 0.15); }
  .bp3-panel-stack2-header > span{
    -webkit-box-align:stretch;
        -ms-flex-align:stretch;
            align-items:stretch;
    display:-webkit-box;
    display:-ms-flexbox;
    display:flex;
    -webkit-box-flex:1;
        -ms-flex:1;
            flex:1; }
  .bp3-panel-stack2-header .bp3-heading{
    margin:0 5px; }

.bp3-button.bp3-panel-stack2-header-back{
  margin-left:5px;
  padding-left:0;
  white-space:nowrap; }
  .bp3-button.bp3-panel-stack2-header-back .bp3-icon{
    margin:0 2px; }

.bp3-panel-stack2-view{
  bottom:0;
  left:0;
  position:absolute;
  right:0;
  top:0;
  background-color:#ffffff;
  border-right:1px solid rgba(16, 22, 26, 0.15);
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-orient:vertical;
  -webkit-box-direction:normal;
      -ms-flex-direction:column;
          flex-direction:column;
  margin-right:-1px;
  overflow-y:auto;
  z-index:1; }
  .bp3-dark .bp3-panel-stack2-view{
    background-color:#30404d; }
  .bp3-panel-stack2-view:nth-last-child(n + 4){
    display:none; }

.bp3-panel-stack2-push .bp3-panel-stack2-enter, .bp3-panel-stack2-push .bp3-panel-stack2-appear{
  -webkit-transform:translateX(100%);
          transform:translateX(100%);
  opacity:0; }

.bp3-panel-stack2-push .bp3-panel-stack2-enter-active, .bp3-panel-stack2-push .bp3-panel-stack2-appear-active{
  -webkit-transform:translate(0%);
          transform:translate(0%);
  opacity:1;
  -webkit-transition-delay:0;
          transition-delay:0;
  -webkit-transition-duration:400ms;
          transition-duration:400ms;
  -webkit-transition-property:opacity, -webkit-transform;
  transition-property:opacity, -webkit-transform;
  transition-property:transform, opacity;
  transition-property:transform, opacity, -webkit-transform;
  -webkit-transition-timing-function:ease;
          transition-timing-function:ease; }

.bp3-panel-stack2-push .bp3-panel-stack2-exit{
  -webkit-transform:translate(0%);
          transform:translate(0%);
  opacity:1; }

.bp3-panel-stack2-push .bp3-panel-stack2-exit-active{
  -webkit-transform:translateX(-50%);
          transform:translateX(-50%);
  opacity:0;
  -webkit-transition-delay:0;
          transition-delay:0;
  -webkit-transition-duration:400ms;
          transition-duration:400ms;
  -webkit-transition-property:opacity, -webkit-transform;
  transition-property:opacity, -webkit-transform;
  transition-property:transform, opacity;
  transition-property:transform, opacity, -webkit-transform;
  -webkit-transition-timing-function:ease;
          transition-timing-function:ease; }

.bp3-panel-stack2-pop .bp3-panel-stack2-enter, .bp3-panel-stack2-pop .bp3-panel-stack2-appear{
  -webkit-transform:translateX(-50%);
          transform:translateX(-50%);
  opacity:0; }

.bp3-panel-stack2-pop .bp3-panel-stack2-enter-active, .bp3-panel-stack2-pop .bp3-panel-stack2-appear-active{
  -webkit-transform:translate(0%);
          transform:translate(0%);
  opacity:1;
  -webkit-transition-delay:0;
          transition-delay:0;
  -webkit-transition-duration:400ms;
          transition-duration:400ms;
  -webkit-transition-property:opacity, -webkit-transform;
  transition-property:opacity, -webkit-transform;
  transition-property:transform, opacity;
  transition-property:transform, opacity, -webkit-transform;
  -webkit-transition-timing-function:ease;
          transition-timing-function:ease; }

.bp3-panel-stack2-pop .bp3-panel-stack2-exit{
  -webkit-transform:translate(0%);
          transform:translate(0%);
  opacity:1; }

.bp3-panel-stack2-pop .bp3-panel-stack2-exit-active{
  -webkit-transform:translateX(100%);
          transform:translateX(100%);
  opacity:0;
  -webkit-transition-delay:0;
          transition-delay:0;
  -webkit-transition-duration:400ms;
          transition-duration:400ms;
  -webkit-transition-property:opacity, -webkit-transform;
  transition-property:opacity, -webkit-transform;
  transition-property:transform, opacity;
  transition-property:transform, opacity, -webkit-transform;
  -webkit-transition-timing-function:ease;
          transition-timing-function:ease; }
.bp3-popover{
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 2px 4px rgba(16, 22, 26, 0.2), 0 8px 24px rgba(16, 22, 26, 0.2);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 2px 4px rgba(16, 22, 26, 0.2), 0 8px 24px rgba(16, 22, 26, 0.2);
  -webkit-transform:scale(1);
          transform:scale(1);
  border-radius:3px;
  display:inline-block;
  z-index:20; }
  .bp3-popover .bp3-popover-arrow{
    height:30px;
    position:absolute;
    width:30px; }
    .bp3-popover .bp3-popover-arrow::before{
      height:20px;
      margin:5px;
      width:20px; }
  .bp3-tether-element-attached-bottom.bp3-tether-target-attached-top > .bp3-popover{
    margin-bottom:17px;
    margin-top:-17px; }
    .bp3-tether-element-attached-bottom.bp3-tether-target-attached-top > .bp3-popover > .bp3-popover-arrow{
      bottom:-11px; }
      .bp3-tether-element-attached-bottom.bp3-tether-target-attached-top > .bp3-popover > .bp3-popover-arrow svg{
        -webkit-transform:rotate(-90deg);
                transform:rotate(-90deg); }
  .bp3-tether-element-attached-left.bp3-tether-target-attached-right > .bp3-popover{
    margin-left:17px; }
    .bp3-tether-element-attached-left.bp3-tether-target-attached-right > .bp3-popover > .bp3-popover-arrow{
      left:-11px; }
      .bp3-tether-element-attached-left.bp3-tether-target-attached-right > .bp3-popover > .bp3-popover-arrow svg{
        -webkit-transform:rotate(0);
                transform:rotate(0); }
  .bp3-tether-element-attached-top.bp3-tether-target-attached-bottom > .bp3-popover{
    margin-top:17px; }
    .bp3-tether-element-attached-top.bp3-tether-target-attached-bottom > .bp3-popover > .bp3-popover-arrow{
      top:-11px; }
      .bp3-tether-element-attached-top.bp3-tether-target-attached-bottom > .bp3-popover > .bp3-popover-arrow svg{
        -webkit-transform:rotate(90deg);
                transform:rotate(90deg); }
  .bp3-tether-element-attached-right.bp3-tether-target-attached-left > .bp3-popover{
    margin-left:-17px;
    margin-right:17px; }
    .bp3-tether-element-attached-right.bp3-tether-target-attached-left > .bp3-popover > .bp3-popover-arrow{
      right:-11px; }
      .bp3-tether-element-attached-right.bp3-tether-target-attached-left > .bp3-popover > .bp3-popover-arrow svg{
        -webkit-transform:rotate(180deg);
                transform:rotate(180deg); }
  .bp3-tether-element-attached-middle > .bp3-popover > .bp3-popover-arrow{
    top:50%;
    -webkit-transform:translateY(-50%);
            transform:translateY(-50%); }
  .bp3-tether-element-attached-center > .bp3-popover > .bp3-popover-arrow{
    right:50%;
    -webkit-transform:translateX(50%);
            transform:translateX(50%); }
  .bp3-tether-element-attached-top.bp3-tether-target-attached-top > .bp3-popover > .bp3-popover-arrow{
    top:-0.3934px; }
  .bp3-tether-element-attached-right.bp3-tether-target-attached-right > .bp3-popover > .bp3-popover-arrow{
    right:-0.3934px; }
  .bp3-tether-element-attached-left.bp3-tether-target-attached-left > .bp3-popover > .bp3-popover-arrow{
    left:-0.3934px; }
  .bp3-tether-element-attached-bottom.bp3-tether-target-attached-bottom > .bp3-popover > .bp3-popover-arrow{
    bottom:-0.3934px; }
  .bp3-tether-element-attached-top.bp3-tether-element-attached-left > .bp3-popover{
    -webkit-transform-origin:top left;
            transform-origin:top left; }
  .bp3-tether-element-attached-top.bp3-tether-element-attached-center > .bp3-popover{
    -webkit-transform-origin:top center;
            transform-origin:top center; }
  .bp3-tether-element-attached-top.bp3-tether-element-attached-right > .bp3-popover{
    -webkit-transform-origin:top right;
            transform-origin:top right; }
  .bp3-tether-element-attached-middle.bp3-tether-element-attached-left > .bp3-popover{
    -webkit-transform-origin:center left;
            transform-origin:center left; }
  .bp3-tether-element-attached-middle.bp3-tether-element-attached-center > .bp3-popover{
    -webkit-transform-origin:center center;
            transform-origin:center center; }
  .bp3-tether-element-attached-middle.bp3-tether-element-attached-right > .bp3-popover{
    -webkit-transform-origin:center right;
            transform-origin:center right; }
  .bp3-tether-element-attached-bottom.bp3-tether-element-attached-left > .bp3-popover{
    -webkit-transform-origin:bottom left;
            transform-origin:bottom left; }
  .bp3-tether-element-attached-bottom.bp3-tether-element-attached-center > .bp3-popover{
    -webkit-transform-origin:bottom center;
            transform-origin:bottom center; }
  .bp3-tether-element-attached-bottom.bp3-tether-element-attached-right > .bp3-popover{
    -webkit-transform-origin:bottom right;
            transform-origin:bottom right; }
  .bp3-popover .bp3-popover-content{
    background:#ffffff;
    color:inherit; }
  .bp3-popover .bp3-popover-arrow::before{
    -webkit-box-shadow:1px 1px 6px rgba(16, 22, 26, 0.2);
            box-shadow:1px 1px 6px rgba(16, 22, 26, 0.2); }
  .bp3-popover .bp3-popover-arrow-border{
    fill:#10161a;
    fill-opacity:0.1; }
  .bp3-popover .bp3-popover-arrow-fill{
    fill:#ffffff; }
  .bp3-popover-enter > .bp3-popover, .bp3-popover-appear > .bp3-popover{
    -webkit-transform:scale(0.3);
            transform:scale(0.3); }
  .bp3-popover-enter-active > .bp3-popover, .bp3-popover-appear-active > .bp3-popover{
    -webkit-transform:scale(1);
            transform:scale(1);
    -webkit-transition-delay:0;
            transition-delay:0;
    -webkit-transition-duration:300ms;
            transition-duration:300ms;
    -webkit-transition-property:-webkit-transform;
    transition-property:-webkit-transform;
    transition-property:transform;
    transition-property:transform, -webkit-transform;
    -webkit-transition-timing-function:cubic-bezier(0.54, 1.12, 0.38, 1.11);
            transition-timing-function:cubic-bezier(0.54, 1.12, 0.38, 1.11); }
  .bp3-popover-exit > .bp3-popover{
    -webkit-transform:scale(1);
            transform:scale(1); }
  .bp3-popover-exit-active > .bp3-popover{
    -webkit-transform:scale(0.3);
            transform:scale(0.3);
    -webkit-transition-delay:0;
            transition-delay:0;
    -webkit-transition-duration:300ms;
            transition-duration:300ms;
    -webkit-transition-property:-webkit-transform;
    transition-property:-webkit-transform;
    transition-property:transform;
    transition-property:transform, -webkit-transform;
    -webkit-transition-timing-function:cubic-bezier(0.54, 1.12, 0.38, 1.11);
            transition-timing-function:cubic-bezier(0.54, 1.12, 0.38, 1.11); }
  .bp3-popover .bp3-popover-content{
    border-radius:3px;
    position:relative; }
  .bp3-popover.bp3-popover-content-sizing .bp3-popover-content{
    max-width:350px;
    padding:20px; }
  .bp3-popover-target + .bp3-overlay .bp3-popover.bp3-popover-content-sizing{
    width:350px; }
  .bp3-popover.bp3-minimal{
    margin:0 !important; }
    .bp3-popover.bp3-minimal .bp3-popover-arrow{
      display:none; }
    .bp3-popover.bp3-minimal.bp3-popover{
      -webkit-transform:scale(1);
              transform:scale(1); }
      .bp3-popover-enter > .bp3-popover.bp3-minimal.bp3-popover, .bp3-popover-appear > .bp3-popover.bp3-minimal.bp3-popover{
        -webkit-transform:scale(1);
                transform:scale(1); }
      .bp3-popover-enter-active > .bp3-popover.bp3-minimal.bp3-popover, .bp3-popover-appear-active > .bp3-popover.bp3-minimal.bp3-popover{
        -webkit-transform:scale(1);
                transform:scale(1);
        -webkit-transition-delay:0;
                transition-delay:0;
        -webkit-transition-duration:100ms;
                transition-duration:100ms;
        -webkit-transition-property:-webkit-transform;
        transition-property:-webkit-transform;
        transition-property:transform;
        transition-property:transform, -webkit-transform;
        -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
                transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
      .bp3-popover-exit > .bp3-popover.bp3-minimal.bp3-popover{
        -webkit-transform:scale(1);
                transform:scale(1); }
      .bp3-popover-exit-active > .bp3-popover.bp3-minimal.bp3-popover{
        -webkit-transform:scale(1);
                transform:scale(1);
        -webkit-transition-delay:0;
                transition-delay:0;
        -webkit-transition-duration:100ms;
                transition-duration:100ms;
        -webkit-transition-property:-webkit-transform;
        transition-property:-webkit-transform;
        transition-property:transform;
        transition-property:transform, -webkit-transform;
        -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
                transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-popover.bp3-dark,
  .bp3-dark .bp3-popover{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 2px 4px rgba(16, 22, 26, 0.4), 0 8px 24px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 2px 4px rgba(16, 22, 26, 0.4), 0 8px 24px rgba(16, 22, 26, 0.4); }
    .bp3-popover.bp3-dark .bp3-popover-content,
    .bp3-dark .bp3-popover .bp3-popover-content{
      background:#30404d;
      color:inherit; }
    .bp3-popover.bp3-dark .bp3-popover-arrow::before,
    .bp3-dark .bp3-popover .bp3-popover-arrow::before{
      -webkit-box-shadow:1px 1px 6px rgba(16, 22, 26, 0.4);
              box-shadow:1px 1px 6px rgba(16, 22, 26, 0.4); }
    .bp3-popover.bp3-dark .bp3-popover-arrow-border,
    .bp3-dark .bp3-popover .bp3-popover-arrow-border{
      fill:#10161a;
      fill-opacity:0.2; }
    .bp3-popover.bp3-dark .bp3-popover-arrow-fill,
    .bp3-dark .bp3-popover .bp3-popover-arrow-fill{
      fill:#30404d; }

.bp3-popover-arrow::before{
  border-radius:2px;
  content:"";
  display:block;
  position:absolute;
  -webkit-transform:rotate(45deg);
          transform:rotate(45deg); }

.bp3-tether-pinned .bp3-popover-arrow{
  display:none; }

.bp3-popover-backdrop{
  background:rgba(255, 255, 255, 0); }

.bp3-transition-container{
  opacity:1;
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  z-index:20; }
  .bp3-transition-container.bp3-popover-enter, .bp3-transition-container.bp3-popover-appear{
    opacity:0; }
  .bp3-transition-container.bp3-popover-enter-active, .bp3-transition-container.bp3-popover-appear-active{
    opacity:1;
    -webkit-transition-delay:0;
            transition-delay:0;
    -webkit-transition-duration:100ms;
            transition-duration:100ms;
    -webkit-transition-property:opacity;
    transition-property:opacity;
    -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
            transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-transition-container.bp3-popover-exit{
    opacity:1; }
  .bp3-transition-container.bp3-popover-exit-active{
    opacity:0;
    -webkit-transition-delay:0;
            transition-delay:0;
    -webkit-transition-duration:100ms;
            transition-duration:100ms;
    -webkit-transition-property:opacity;
    transition-property:opacity;
    -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
            transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-transition-container:focus{
    outline:none; }
  .bp3-transition-container.bp3-popover-leave .bp3-popover-content{
    pointer-events:none; }
  .bp3-transition-container[data-x-out-of-boundaries]{
    display:none; }

span.bp3-popover-target{
  display:inline-block; }

.bp3-popover-wrapper.bp3-fill{
  width:100%; }

.bp3-portal{
  left:0;
  position:absolute;
  right:0;
  top:0; }
@-webkit-keyframes linear-progress-bar-stripes{
  from{
    background-position:0 0; }
  to{
    background-position:30px 0; } }
@keyframes linear-progress-bar-stripes{
  from{
    background-position:0 0; }
  to{
    background-position:30px 0; } }

.bp3-progress-bar{
  background:rgba(92, 112, 128, 0.2);
  border-radius:40px;
  display:block;
  height:8px;
  overflow:hidden;
  position:relative;
  width:100%; }
  .bp3-progress-bar .bp3-progress-meter{
    background:linear-gradient(-45deg, rgba(255, 255, 255, 0.2) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.2) 50%, rgba(255, 255, 255, 0.2) 75%, transparent 75%);
    background-color:rgba(92, 112, 128, 0.8);
    background-size:30px 30px;
    border-radius:40px;
    height:100%;
    position:absolute;
    -webkit-transition:width 200ms cubic-bezier(0.4, 1, 0.75, 0.9);
    transition:width 200ms cubic-bezier(0.4, 1, 0.75, 0.9);
    width:100%; }
  .bp3-progress-bar:not(.bp3-no-animation):not(.bp3-no-stripes) .bp3-progress-meter{
    animation:linear-progress-bar-stripes 300ms linear infinite reverse; }
  .bp3-progress-bar.bp3-no-stripes .bp3-progress-meter{
    background-image:none; }

.bp3-dark .bp3-progress-bar{
  background:rgba(16, 22, 26, 0.5); }
  .bp3-dark .bp3-progress-bar .bp3-progress-meter{
    background-color:#8a9ba8; }

.bp3-progress-bar.bp3-intent-primary .bp3-progress-meter{
  background-color:#137cbd; }

.bp3-progress-bar.bp3-intent-success .bp3-progress-meter{
  background-color:#0f9960; }

.bp3-progress-bar.bp3-intent-warning .bp3-progress-meter{
  background-color:#d9822b; }

.bp3-progress-bar.bp3-intent-danger .bp3-progress-meter{
  background-color:#db3737; }
@-webkit-keyframes skeleton-glow{
  from{
    background:rgba(206, 217, 224, 0.2);
    border-color:rgba(206, 217, 224, 0.2); }
  to{
    background:rgba(92, 112, 128, 0.2);
    border-color:rgba(92, 112, 128, 0.2); } }
@keyframes skeleton-glow{
  from{
    background:rgba(206, 217, 224, 0.2);
    border-color:rgba(206, 217, 224, 0.2); }
  to{
    background:rgba(92, 112, 128, 0.2);
    border-color:rgba(92, 112, 128, 0.2); } }
.bp3-skeleton{
  -webkit-animation:1000ms linear infinite alternate skeleton-glow;
          animation:1000ms linear infinite alternate skeleton-glow;
  background:rgba(206, 217, 224, 0.2);
  background-clip:padding-box !important;
  border-color:rgba(206, 217, 224, 0.2) !important;
  border-radius:2px;
  -webkit-box-shadow:none !important;
          box-shadow:none !important;
  color:transparent !important;
  cursor:default;
  pointer-events:none;
  -webkit-user-select:none;
     -moz-user-select:none;
      -ms-user-select:none;
          user-select:none; }
  .bp3-skeleton::before, .bp3-skeleton::after,
  .bp3-skeleton *{
    visibility:hidden !important; }
.bp3-slider{
  height:40px;
  min-width:150px;
  width:100%;
  cursor:default;
  outline:none;
  position:relative;
  -webkit-user-select:none;
     -moz-user-select:none;
      -ms-user-select:none;
          user-select:none; }
  .bp3-slider:hover{
    cursor:pointer; }
  .bp3-slider:active{
    cursor:-webkit-grabbing;
    cursor:grabbing; }
  .bp3-slider.bp3-disabled{
    cursor:not-allowed;
    opacity:0.5; }
  .bp3-slider.bp3-slider-unlabeled{
    height:16px; }

.bp3-slider-track,
.bp3-slider-progress{
  height:6px;
  left:0;
  right:0;
  top:5px;
  position:absolute; }

.bp3-slider-track{
  border-radius:3px;
  overflow:hidden; }

.bp3-slider-progress{
  background:rgba(92, 112, 128, 0.2); }
  .bp3-dark .bp3-slider-progress{
    background:rgba(16, 22, 26, 0.5); }
  .bp3-slider-progress.bp3-intent-primary{
    background-color:#137cbd; }
  .bp3-slider-progress.bp3-intent-success{
    background-color:#0f9960; }
  .bp3-slider-progress.bp3-intent-warning{
    background-color:#d9822b; }
  .bp3-slider-progress.bp3-intent-danger{
    background-color:#db3737; }

.bp3-slider-handle{
  background-color:#f5f8fa;
  background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.8)), to(rgba(255, 255, 255, 0)));
  background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.8), rgba(255, 255, 255, 0));
  -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
          box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
  color:#182026;
  border-radius:3px;
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 1px 1px rgba(16, 22, 26, 0.2);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 1px 1px rgba(16, 22, 26, 0.2);
  cursor:pointer;
  height:16px;
  left:0;
  position:absolute;
  top:0;
  width:16px; }
  .bp3-slider-handle:hover{
    background-clip:padding-box;
    background-color:#ebf1f5;
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1); }
  .bp3-slider-handle:active, .bp3-slider-handle.bp3-active{
    background-color:#d8e1e8;
    background-image:none;
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 2px rgba(16, 22, 26, 0.2);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
  .bp3-slider-handle:disabled, .bp3-slider-handle.bp3-disabled{
    background-color:rgba(206, 217, 224, 0.5);
    background-image:none;
    -webkit-box-shadow:none;
            box-shadow:none;
    color:rgba(92, 112, 128, 0.6);
    cursor:not-allowed;
    outline:none; }
    .bp3-slider-handle:disabled.bp3-active, .bp3-slider-handle:disabled.bp3-active:hover, .bp3-slider-handle.bp3-disabled.bp3-active, .bp3-slider-handle.bp3-disabled.bp3-active:hover{
      background:rgba(206, 217, 224, 0.7); }
  .bp3-slider-handle:focus{
    z-index:1; }
  .bp3-slider-handle:hover{
    background-clip:padding-box;
    background-color:#ebf1f5;
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 1px 1px rgba(16, 22, 26, 0.2);
    cursor:-webkit-grab;
    cursor:grab;
    z-index:2; }
  .bp3-slider-handle.bp3-active{
    background-color:#d8e1e8;
    background-image:none;
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 2px rgba(16, 22, 26, 0.2);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 2px rgba(16, 22, 26, 0.2);
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 1px rgba(16, 22, 26, 0.1);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 1px rgba(16, 22, 26, 0.1);
    cursor:-webkit-grabbing;
    cursor:grabbing; }
  .bp3-disabled .bp3-slider-handle{
    background:#bfccd6;
    -webkit-box-shadow:none;
            box-shadow:none;
    pointer-events:none; }
  .bp3-dark .bp3-slider-handle{
    background-color:#394b59;
    background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.05)), to(rgba(255, 255, 255, 0)));
    background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0));
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
    color:#f5f8fa; }
    .bp3-dark .bp3-slider-handle:hover, .bp3-dark .bp3-slider-handle:active, .bp3-dark .bp3-slider-handle.bp3-active{
      color:#f5f8fa; }
    .bp3-dark .bp3-slider-handle:hover{
      background-color:#30404d;
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4); }
    .bp3-dark .bp3-slider-handle:active, .bp3-dark .bp3-slider-handle.bp3-active{
      background-color:#202b33;
      background-image:none;
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.6), inset 0 1px 2px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.6), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
    .bp3-dark .bp3-slider-handle:disabled, .bp3-dark .bp3-slider-handle.bp3-disabled{
      background-color:rgba(57, 75, 89, 0.5);
      background-image:none;
      -webkit-box-shadow:none;
              box-shadow:none;
      color:rgba(167, 182, 194, 0.6); }
      .bp3-dark .bp3-slider-handle:disabled.bp3-active, .bp3-dark .bp3-slider-handle.bp3-disabled.bp3-active{
        background:rgba(57, 75, 89, 0.7); }
    .bp3-dark .bp3-slider-handle .bp3-button-spinner .bp3-spinner-head{
      background:rgba(16, 22, 26, 0.5);
      stroke:#8a9ba8; }
    .bp3-dark .bp3-slider-handle, .bp3-dark .bp3-slider-handle:hover{
      background-color:#394b59; }
    .bp3-dark .bp3-slider-handle.bp3-active{
      background-color:#293742; }
  .bp3-dark .bp3-disabled .bp3-slider-handle{
    background:#5c7080;
    border-color:#5c7080;
    -webkit-box-shadow:none;
            box-shadow:none; }
  .bp3-slider-handle .bp3-slider-label{
    background:#394b59;
    border-radius:3px;
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 2px 4px rgba(16, 22, 26, 0.2), 0 8px 24px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 2px 4px rgba(16, 22, 26, 0.2), 0 8px 24px rgba(16, 22, 26, 0.2);
    color:#f5f8fa;
    margin-left:8px; }
    .bp3-dark .bp3-slider-handle .bp3-slider-label{
      background:#e1e8ed;
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 2px 4px rgba(16, 22, 26, 0.4), 0 8px 24px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 2px 4px rgba(16, 22, 26, 0.4), 0 8px 24px rgba(16, 22, 26, 0.4);
      color:#394b59; }
    .bp3-disabled .bp3-slider-handle .bp3-slider-label{
      -webkit-box-shadow:none;
              box-shadow:none; }
  .bp3-slider-handle.bp3-start, .bp3-slider-handle.bp3-end{
    width:8px; }
  .bp3-slider-handle.bp3-start{
    border-bottom-right-radius:0;
    border-top-right-radius:0; }
  .bp3-slider-handle.bp3-end{
    border-bottom-left-radius:0;
    border-top-left-radius:0;
    margin-left:8px; }
    .bp3-slider-handle.bp3-end .bp3-slider-label{
      margin-left:0; }

.bp3-slider-label{
  -webkit-transform:translate(-50%, 20px);
          transform:translate(-50%, 20px);
  display:inline-block;
  font-size:12px;
  line-height:1;
  padding:2px 5px;
  position:absolute;
  vertical-align:top; }

.bp3-slider.bp3-vertical{
  height:150px;
  min-width:40px;
  width:40px; }
  .bp3-slider.bp3-vertical .bp3-slider-track,
  .bp3-slider.bp3-vertical .bp3-slider-progress{
    bottom:0;
    height:auto;
    left:5px;
    top:0;
    width:6px; }
  .bp3-slider.bp3-vertical .bp3-slider-progress{
    top:auto; }
  .bp3-slider.bp3-vertical .bp3-slider-label{
    -webkit-transform:translate(20px, 50%);
            transform:translate(20px, 50%); }
  .bp3-slider.bp3-vertical .bp3-slider-handle{
    top:auto; }
    .bp3-slider.bp3-vertical .bp3-slider-handle .bp3-slider-label{
      margin-left:0;
      margin-top:-8px; }
    .bp3-slider.bp3-vertical .bp3-slider-handle.bp3-end, .bp3-slider.bp3-vertical .bp3-slider-handle.bp3-start{
      height:8px;
      margin-left:0;
      width:16px; }
    .bp3-slider.bp3-vertical .bp3-slider-handle.bp3-start{
      border-bottom-right-radius:3px;
      border-top-left-radius:0; }
      .bp3-slider.bp3-vertical .bp3-slider-handle.bp3-start .bp3-slider-label{
        -webkit-transform:translate(20px);
                transform:translate(20px); }
    .bp3-slider.bp3-vertical .bp3-slider-handle.bp3-end{
      border-bottom-left-radius:0;
      border-bottom-right-radius:0;
      border-top-left-radius:3px;
      margin-bottom:8px; }

@-webkit-keyframes pt-spinner-animation{
  from{
    -webkit-transform:rotate(0deg);
            transform:rotate(0deg); }
  to{
    -webkit-transform:rotate(360deg);
            transform:rotate(360deg); } }

@keyframes pt-spinner-animation{
  from{
    -webkit-transform:rotate(0deg);
            transform:rotate(0deg); }
  to{
    -webkit-transform:rotate(360deg);
            transform:rotate(360deg); } }

.bp3-spinner{
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-pack:center;
      -ms-flex-pack:center;
          justify-content:center;
  overflow:visible;
  vertical-align:middle; }
  .bp3-spinner svg{
    display:block; }
  .bp3-spinner path{
    fill-opacity:0; }
  .bp3-spinner .bp3-spinner-head{
    stroke:rgba(92, 112, 128, 0.8);
    stroke-linecap:round;
    -webkit-transform-origin:center;
            transform-origin:center;
    -webkit-transition:stroke-dashoffset 200ms cubic-bezier(0.4, 1, 0.75, 0.9);
    transition:stroke-dashoffset 200ms cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-spinner .bp3-spinner-track{
    stroke:rgba(92, 112, 128, 0.2); }

.bp3-spinner-animation{
  -webkit-animation:pt-spinner-animation 500ms linear infinite;
          animation:pt-spinner-animation 500ms linear infinite; }
  .bp3-no-spin > .bp3-spinner-animation{
    -webkit-animation:none;
            animation:none; }

.bp3-dark .bp3-spinner .bp3-spinner-head{
  stroke:#8a9ba8; }

.bp3-dark .bp3-spinner .bp3-spinner-track{
  stroke:rgba(16, 22, 26, 0.5); }

.bp3-spinner.bp3-intent-primary .bp3-spinner-head{
  stroke:#137cbd; }

.bp3-spinner.bp3-intent-success .bp3-spinner-head{
  stroke:#0f9960; }

.bp3-spinner.bp3-intent-warning .bp3-spinner-head{
  stroke:#d9822b; }

.bp3-spinner.bp3-intent-danger .bp3-spinner-head{
  stroke:#db3737; }
.bp3-tabs.bp3-vertical{
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex; }
  .bp3-tabs.bp3-vertical > .bp3-tab-list{
    -webkit-box-align:start;
        -ms-flex-align:start;
            align-items:flex-start;
    -webkit-box-orient:vertical;
    -webkit-box-direction:normal;
        -ms-flex-direction:column;
            flex-direction:column; }
    .bp3-tabs.bp3-vertical > .bp3-tab-list .bp3-tab{
      border-radius:3px;
      padding:0 10px;
      width:100%; }
      .bp3-tabs.bp3-vertical > .bp3-tab-list .bp3-tab[aria-selected="true"]{
        background-color:rgba(19, 124, 189, 0.2);
        -webkit-box-shadow:none;
                box-shadow:none; }
    .bp3-tabs.bp3-vertical > .bp3-tab-list .bp3-tab-indicator-wrapper .bp3-tab-indicator{
      background-color:rgba(19, 124, 189, 0.2);
      border-radius:3px;
      bottom:0;
      height:auto;
      left:0;
      right:0;
      top:0; }
  .bp3-tabs.bp3-vertical > .bp3-tab-panel{
    margin-top:0;
    padding-left:20px; }

.bp3-tab-list{
  -webkit-box-align:end;
      -ms-flex-align:end;
          align-items:flex-end;
  border:none;
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-flex:0;
      -ms-flex:0 0 auto;
          flex:0 0 auto;
  list-style:none;
  margin:0;
  padding:0;
  position:relative; }
  .bp3-tab-list > *:not(:last-child){
    margin-right:20px; }

.bp3-tab{
  overflow:hidden;
  text-overflow:ellipsis;
  white-space:nowrap;
  word-wrap:normal;
  color:#182026;
  cursor:pointer;
  -webkit-box-flex:0;
      -ms-flex:0 0 auto;
          flex:0 0 auto;
  font-size:14px;
  line-height:30px;
  max-width:100%;
  position:relative;
  vertical-align:top; }
  .bp3-tab a{
    color:inherit;
    display:block;
    text-decoration:none; }
  .bp3-tab-indicator-wrapper ~ .bp3-tab{
    background-color:transparent !important;
    -webkit-box-shadow:none !important;
            box-shadow:none !important; }
  .bp3-tab[aria-disabled="true"]{
    color:rgba(92, 112, 128, 0.6);
    cursor:not-allowed; }
  .bp3-tab[aria-selected="true"]{
    border-radius:0;
    -webkit-box-shadow:inset 0 -3px 0 #106ba3;
            box-shadow:inset 0 -3px 0 #106ba3; }
  .bp3-tab[aria-selected="true"], .bp3-tab:not([aria-disabled="true"]):hover{
    color:#106ba3; }
  .bp3-tab:focus{
    -moz-outline-radius:0; }
  .bp3-large > .bp3-tab{
    font-size:16px;
    line-height:40px; }

.bp3-tab-panel{
  margin-top:20px; }
  .bp3-tab-panel[aria-hidden="true"]{
    display:none; }

.bp3-tab-indicator-wrapper{
  left:0;
  pointer-events:none;
  position:absolute;
  top:0;
  -webkit-transform:translateX(0), translateY(0);
          transform:translateX(0), translateY(0);
  -webkit-transition:height, width, -webkit-transform;
  transition:height, width, -webkit-transform;
  transition:height, transform, width;
  transition:height, transform, width, -webkit-transform;
  -webkit-transition-duration:200ms;
          transition-duration:200ms;
  -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
          transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-tab-indicator-wrapper .bp3-tab-indicator{
    background-color:#106ba3;
    bottom:0;
    height:3px;
    left:0;
    position:absolute;
    right:0; }
  .bp3-tab-indicator-wrapper.bp3-no-animation{
    -webkit-transition:none;
    transition:none; }

.bp3-dark .bp3-tab{
  color:#f5f8fa; }
  .bp3-dark .bp3-tab[aria-disabled="true"]{
    color:rgba(167, 182, 194, 0.6); }
  .bp3-dark .bp3-tab[aria-selected="true"]{
    -webkit-box-shadow:inset 0 -3px 0 #48aff0;
            box-shadow:inset 0 -3px 0 #48aff0; }
  .bp3-dark .bp3-tab[aria-selected="true"], .bp3-dark .bp3-tab:not([aria-disabled="true"]):hover{
    color:#48aff0; }

.bp3-dark .bp3-tab-indicator{
  background-color:#48aff0; }

.bp3-flex-expander{
  -webkit-box-flex:1;
      -ms-flex:1 1;
          flex:1 1; }
.bp3-tag{
  display:-webkit-inline-box;
  display:-ms-inline-flexbox;
  display:inline-flex;
  -webkit-box-orient:horizontal;
  -webkit-box-direction:normal;
      -ms-flex-direction:row;
          flex-direction:row;
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  background-color:#5c7080;
  border:none;
  border-radius:3px;
  -webkit-box-shadow:none;
          box-shadow:none;
  color:#f5f8fa;
  font-size:12px;
  line-height:16px;
  max-width:100%;
  min-height:20px;
  min-width:20px;
  padding:2px 6px;
  position:relative; }
  .bp3-tag.bp3-interactive{
    cursor:pointer; }
    .bp3-tag.bp3-interactive:hover{
      background-color:rgba(92, 112, 128, 0.85); }
    .bp3-tag.bp3-interactive.bp3-active, .bp3-tag.bp3-interactive:active{
      background-color:rgba(92, 112, 128, 0.7); }
  .bp3-tag > *{
    -webkit-box-flex:0;
        -ms-flex-positive:0;
            flex-grow:0;
    -ms-flex-negative:0;
        flex-shrink:0; }
  .bp3-tag > .bp3-fill{
    -webkit-box-flex:1;
        -ms-flex-positive:1;
            flex-grow:1;
    -ms-flex-negative:1;
        flex-shrink:1; }
  .bp3-tag::before,
  .bp3-tag > *{
    margin-right:4px; }
  .bp3-tag:empty::before,
  .bp3-tag > :last-child{
    margin-right:0; }
  .bp3-tag:focus{
    outline:rgba(19, 124, 189, 0.6) auto 2px;
    outline-offset:0;
    -moz-outline-radius:6px; }
  .bp3-tag.bp3-round{
    border-radius:30px;
    padding-left:8px;
    padding-right:8px; }
  .bp3-dark .bp3-tag{
    background-color:#bfccd6;
    color:#182026; }
    .bp3-dark .bp3-tag.bp3-interactive{
      cursor:pointer; }
      .bp3-dark .bp3-tag.bp3-interactive:hover{
        background-color:rgba(191, 204, 214, 0.85); }
      .bp3-dark .bp3-tag.bp3-interactive.bp3-active, .bp3-dark .bp3-tag.bp3-interactive:active{
        background-color:rgba(191, 204, 214, 0.7); }
    .bp3-dark .bp3-tag > .bp3-icon, .bp3-dark .bp3-tag .bp3-icon-standard, .bp3-dark .bp3-tag .bp3-icon-large{
      fill:currentColor; }
  .bp3-tag > .bp3-icon, .bp3-tag .bp3-icon-standard, .bp3-tag .bp3-icon-large{
    fill:#ffffff; }
  .bp3-tag.bp3-large,
  .bp3-large .bp3-tag{
    font-size:14px;
    line-height:20px;
    min-height:30px;
    min-width:30px;
    padding:5px 10px; }
    .bp3-tag.bp3-large::before,
    .bp3-tag.bp3-large > *,
    .bp3-large .bp3-tag::before,
    .bp3-large .bp3-tag > *{
      margin-right:7px; }
    .bp3-tag.bp3-large:empty::before,
    .bp3-tag.bp3-large > :last-child,
    .bp3-large .bp3-tag:empty::before,
    .bp3-large .bp3-tag > :last-child{
      margin-right:0; }
    .bp3-tag.bp3-large.bp3-round,
    .bp3-large .bp3-tag.bp3-round{
      padding-left:12px;
      padding-right:12px; }
  .bp3-tag.bp3-intent-primary{
    background:#137cbd;
    color:#ffffff; }
    .bp3-tag.bp3-intent-primary.bp3-interactive{
      cursor:pointer; }
      .bp3-tag.bp3-intent-primary.bp3-interactive:hover{
        background-color:rgba(19, 124, 189, 0.85); }
      .bp3-tag.bp3-intent-primary.bp3-interactive.bp3-active, .bp3-tag.bp3-intent-primary.bp3-interactive:active{
        background-color:rgba(19, 124, 189, 0.7); }
  .bp3-tag.bp3-intent-success{
    background:#0f9960;
    color:#ffffff; }
    .bp3-tag.bp3-intent-success.bp3-interactive{
      cursor:pointer; }
      .bp3-tag.bp3-intent-success.bp3-interactive:hover{
        background-color:rgba(15, 153, 96, 0.85); }
      .bp3-tag.bp3-intent-success.bp3-interactive.bp3-active, .bp3-tag.bp3-intent-success.bp3-interactive:active{
        background-color:rgba(15, 153, 96, 0.7); }
  .bp3-tag.bp3-intent-warning{
    background:#d9822b;
    color:#ffffff; }
    .bp3-tag.bp3-intent-warning.bp3-interactive{
      cursor:pointer; }
      .bp3-tag.bp3-intent-warning.bp3-interactive:hover{
        background-color:rgba(217, 130, 43, 0.85); }
      .bp3-tag.bp3-intent-warning.bp3-interactive.bp3-active, .bp3-tag.bp3-intent-warning.bp3-interactive:active{
        background-color:rgba(217, 130, 43, 0.7); }
  .bp3-tag.bp3-intent-danger{
    background:#db3737;
    color:#ffffff; }
    .bp3-tag.bp3-intent-danger.bp3-interactive{
      cursor:pointer; }
      .bp3-tag.bp3-intent-danger.bp3-interactive:hover{
        background-color:rgba(219, 55, 55, 0.85); }
      .bp3-tag.bp3-intent-danger.bp3-interactive.bp3-active, .bp3-tag.bp3-intent-danger.bp3-interactive:active{
        background-color:rgba(219, 55, 55, 0.7); }
  .bp3-tag.bp3-fill{
    display:-webkit-box;
    display:-ms-flexbox;
    display:flex;
    width:100%; }
  .bp3-tag.bp3-minimal > .bp3-icon, .bp3-tag.bp3-minimal .bp3-icon-standard, .bp3-tag.bp3-minimal .bp3-icon-large{
    fill:#5c7080; }
  .bp3-tag.bp3-minimal:not([class*="bp3-intent-"]){
    background-color:rgba(138, 155, 168, 0.2);
    color:#182026; }
    .bp3-tag.bp3-minimal:not([class*="bp3-intent-"]).bp3-interactive{
      cursor:pointer; }
      .bp3-tag.bp3-minimal:not([class*="bp3-intent-"]).bp3-interactive:hover{
        background-color:rgba(92, 112, 128, 0.3); }
      .bp3-tag.bp3-minimal:not([class*="bp3-intent-"]).bp3-interactive.bp3-active, .bp3-tag.bp3-minimal:not([class*="bp3-intent-"]).bp3-interactive:active{
        background-color:rgba(92, 112, 128, 0.4); }
    .bp3-dark .bp3-tag.bp3-minimal:not([class*="bp3-intent-"]){
      color:#f5f8fa; }
      .bp3-dark .bp3-tag.bp3-minimal:not([class*="bp3-intent-"]).bp3-interactive{
        cursor:pointer; }
        .bp3-dark .bp3-tag.bp3-minimal:not([class*="bp3-intent-"]).bp3-interactive:hover{
          background-color:rgba(191, 204, 214, 0.3); }
        .bp3-dark .bp3-tag.bp3-minimal:not([class*="bp3-intent-"]).bp3-interactive.bp3-active, .bp3-dark .bp3-tag.bp3-minimal:not([class*="bp3-intent-"]).bp3-interactive:active{
          background-color:rgba(191, 204, 214, 0.4); }
      .bp3-dark .bp3-tag.bp3-minimal:not([class*="bp3-intent-"]) > .bp3-icon, .bp3-dark .bp3-tag.bp3-minimal:not([class*="bp3-intent-"]) .bp3-icon-standard, .bp3-dark .bp3-tag.bp3-minimal:not([class*="bp3-intent-"]) .bp3-icon-large{
        fill:#a7b6c2; }
  .bp3-tag.bp3-minimal.bp3-intent-primary{
    background-color:rgba(19, 124, 189, 0.15);
    color:#106ba3; }
    .bp3-tag.bp3-minimal.bp3-intent-primary.bp3-interactive{
      cursor:pointer; }
      .bp3-tag.bp3-minimal.bp3-intent-primary.bp3-interactive:hover{
        background-color:rgba(19, 124, 189, 0.25); }
      .bp3-tag.bp3-minimal.bp3-intent-primary.bp3-interactive.bp3-active, .bp3-tag.bp3-minimal.bp3-intent-primary.bp3-interactive:active{
        background-color:rgba(19, 124, 189, 0.35); }
    .bp3-tag.bp3-minimal.bp3-intent-primary > .bp3-icon, .bp3-tag.bp3-minimal.bp3-intent-primary .bp3-icon-standard, .bp3-tag.bp3-minimal.bp3-intent-primary .bp3-icon-large{
      fill:#137cbd; }
    .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-primary{
      background-color:rgba(19, 124, 189, 0.25);
      color:#48aff0; }
      .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-primary.bp3-interactive{
        cursor:pointer; }
        .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-primary.bp3-interactive:hover{
          background-color:rgba(19, 124, 189, 0.35); }
        .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-primary.bp3-interactive.bp3-active, .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-primary.bp3-interactive:active{
          background-color:rgba(19, 124, 189, 0.45); }
  .bp3-tag.bp3-minimal.bp3-intent-success{
    background-color:rgba(15, 153, 96, 0.15);
    color:#0d8050; }
    .bp3-tag.bp3-minimal.bp3-intent-success.bp3-interactive{
      cursor:pointer; }
      .bp3-tag.bp3-minimal.bp3-intent-success.bp3-interactive:hover{
        background-color:rgba(15, 153, 96, 0.25); }
      .bp3-tag.bp3-minimal.bp3-intent-success.bp3-interactive.bp3-active, .bp3-tag.bp3-minimal.bp3-intent-success.bp3-interactive:active{
        background-color:rgba(15, 153, 96, 0.35); }
    .bp3-tag.bp3-minimal.bp3-intent-success > .bp3-icon, .bp3-tag.bp3-minimal.bp3-intent-success .bp3-icon-standard, .bp3-tag.bp3-minimal.bp3-intent-success .bp3-icon-large{
      fill:#0f9960; }
    .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-success{
      background-color:rgba(15, 153, 96, 0.25);
      color:#3dcc91; }
      .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-success.bp3-interactive{
        cursor:pointer; }
        .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-success.bp3-interactive:hover{
          background-color:rgba(15, 153, 96, 0.35); }
        .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-success.bp3-interactive.bp3-active, .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-success.bp3-interactive:active{
          background-color:rgba(15, 153, 96, 0.45); }
  .bp3-tag.bp3-minimal.bp3-intent-warning{
    background-color:rgba(217, 130, 43, 0.15);
    color:#bf7326; }
    .bp3-tag.bp3-minimal.bp3-intent-warning.bp3-interactive{
      cursor:pointer; }
      .bp3-tag.bp3-minimal.bp3-intent-warning.bp3-interactive:hover{
        background-color:rgba(217, 130, 43, 0.25); }
      .bp3-tag.bp3-minimal.bp3-intent-warning.bp3-interactive.bp3-active, .bp3-tag.bp3-minimal.bp3-intent-warning.bp3-interactive:active{
        background-color:rgba(217, 130, 43, 0.35); }
    .bp3-tag.bp3-minimal.bp3-intent-warning > .bp3-icon, .bp3-tag.bp3-minimal.bp3-intent-warning .bp3-icon-standard, .bp3-tag.bp3-minimal.bp3-intent-warning .bp3-icon-large{
      fill:#d9822b; }
    .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-warning{
      background-color:rgba(217, 130, 43, 0.25);
      color:#ffb366; }
      .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-warning.bp3-interactive{
        cursor:pointer; }
        .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-warning.bp3-interactive:hover{
          background-color:rgba(217, 130, 43, 0.35); }
        .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-warning.bp3-interactive.bp3-active, .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-warning.bp3-interactive:active{
          background-color:rgba(217, 130, 43, 0.45); }
  .bp3-tag.bp3-minimal.bp3-intent-danger{
    background-color:rgba(219, 55, 55, 0.15);
    color:#c23030; }
    .bp3-tag.bp3-minimal.bp3-intent-danger.bp3-interactive{
      cursor:pointer; }
      .bp3-tag.bp3-minimal.bp3-intent-danger.bp3-interactive:hover{
        background-color:rgba(219, 55, 55, 0.25); }
      .bp3-tag.bp3-minimal.bp3-intent-danger.bp3-interactive.bp3-active, .bp3-tag.bp3-minimal.bp3-intent-danger.bp3-interactive:active{
        background-color:rgba(219, 55, 55, 0.35); }
    .bp3-tag.bp3-minimal.bp3-intent-danger > .bp3-icon, .bp3-tag.bp3-minimal.bp3-intent-danger .bp3-icon-standard, .bp3-tag.bp3-minimal.bp3-intent-danger .bp3-icon-large{
      fill:#db3737; }
    .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-danger{
      background-color:rgba(219, 55, 55, 0.25);
      color:#ff7373; }
      .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-danger.bp3-interactive{
        cursor:pointer; }
        .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-danger.bp3-interactive:hover{
          background-color:rgba(219, 55, 55, 0.35); }
        .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-danger.bp3-interactive.bp3-active, .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-danger.bp3-interactive:active{
          background-color:rgba(219, 55, 55, 0.45); }

.bp3-tag-remove{
  background:none;
  border:none;
  color:inherit;
  cursor:pointer;
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  margin-bottom:-2px;
  margin-right:-6px !important;
  margin-top:-2px;
  opacity:0.5;
  padding:2px;
  padding-left:0; }
  .bp3-tag-remove:hover{
    background:none;
    opacity:0.8;
    text-decoration:none; }
  .bp3-tag-remove:active{
    opacity:1; }
  .bp3-tag-remove:empty::before{
    font-family:"Icons16", sans-serif;
    font-size:16px;
    font-style:normal;
    font-weight:400;
    line-height:1;
    -moz-osx-font-smoothing:grayscale;
    -webkit-font-smoothing:antialiased;
    content:""; }
  .bp3-large .bp3-tag-remove{
    margin-right:-10px !important;
    padding:0 5px 0 0; }
    .bp3-large .bp3-tag-remove:empty::before{
      font-family:"Icons20", sans-serif;
      font-size:20px;
      font-style:normal;
      font-weight:400;
      line-height:1; }
.bp3-tag-input{
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-orient:horizontal;
  -webkit-box-direction:normal;
      -ms-flex-direction:row;
          flex-direction:row;
  -webkit-box-align:start;
      -ms-flex-align:start;
          align-items:flex-start;
  cursor:text;
  height:auto;
  line-height:inherit;
  min-height:30px;
  padding-left:5px;
  padding-right:0; }
  .bp3-tag-input > *{
    -webkit-box-flex:0;
        -ms-flex-positive:0;
            flex-grow:0;
    -ms-flex-negative:0;
        flex-shrink:0; }
  .bp3-tag-input > .bp3-tag-input-values{
    -webkit-box-flex:1;
        -ms-flex-positive:1;
            flex-grow:1;
    -ms-flex-negative:1;
        flex-shrink:1; }
  .bp3-tag-input .bp3-tag-input-icon{
    color:#5c7080;
    margin-left:2px;
    margin-right:7px;
    margin-top:7px; }
  .bp3-tag-input .bp3-tag-input-values{
    display:-webkit-box;
    display:-ms-flexbox;
    display:flex;
    -webkit-box-orient:horizontal;
    -webkit-box-direction:normal;
        -ms-flex-direction:row;
            flex-direction:row;
    -webkit-box-align:center;
        -ms-flex-align:center;
            align-items:center;
    -ms-flex-item-align:stretch;
        align-self:stretch;
    -ms-flex-wrap:wrap;
        flex-wrap:wrap;
    margin-right:7px;
    margin-top:5px;
    min-width:0; }
    .bp3-tag-input .bp3-tag-input-values > *{
      -webkit-box-flex:0;
          -ms-flex-positive:0;
              flex-grow:0;
      -ms-flex-negative:0;
          flex-shrink:0; }
    .bp3-tag-input .bp3-tag-input-values > .bp3-fill{
      -webkit-box-flex:1;
          -ms-flex-positive:1;
              flex-grow:1;
      -ms-flex-negative:1;
          flex-shrink:1; }
    .bp3-tag-input .bp3-tag-input-values::before,
    .bp3-tag-input .bp3-tag-input-values > *{
      margin-right:5px; }
    .bp3-tag-input .bp3-tag-input-values:empty::before,
    .bp3-tag-input .bp3-tag-input-values > :last-child{
      margin-right:0; }
    .bp3-tag-input .bp3-tag-input-values:first-child .bp3-input-ghost:first-child{
      padding-left:5px; }
    .bp3-tag-input .bp3-tag-input-values > *{
      margin-bottom:5px; }
  .bp3-tag-input .bp3-tag{
    overflow-wrap:break-word; }
    .bp3-tag-input .bp3-tag.bp3-active{
      outline:rgba(19, 124, 189, 0.6) auto 2px;
      outline-offset:0;
      -moz-outline-radius:6px; }
  .bp3-tag-input .bp3-input-ghost{
    -webkit-box-flex:1;
        -ms-flex:1 1 auto;
            flex:1 1 auto;
    line-height:20px;
    width:80px; }
    .bp3-tag-input .bp3-input-ghost:disabled, .bp3-tag-input .bp3-input-ghost.bp3-disabled{
      cursor:not-allowed; }
  .bp3-tag-input .bp3-button,
  .bp3-tag-input .bp3-spinner{
    margin:3px;
    margin-left:0; }
  .bp3-tag-input .bp3-button{
    min-height:24px;
    min-width:24px;
    padding:0 7px; }
  .bp3-tag-input.bp3-large{
    height:auto;
    min-height:40px; }
    .bp3-tag-input.bp3-large::before,
    .bp3-tag-input.bp3-large > *{
      margin-right:10px; }
    .bp3-tag-input.bp3-large:empty::before,
    .bp3-tag-input.bp3-large > :last-child{
      margin-right:0; }
    .bp3-tag-input.bp3-large .bp3-tag-input-icon{
      margin-left:5px;
      margin-top:10px; }
    .bp3-tag-input.bp3-large .bp3-input-ghost{
      line-height:30px; }
    .bp3-tag-input.bp3-large .bp3-button{
      min-height:30px;
      min-width:30px;
      padding:5px 10px;
      margin:5px;
      margin-left:0; }
    .bp3-tag-input.bp3-large .bp3-spinner{
      margin:8px;
      margin-left:0; }
  .bp3-tag-input.bp3-active{
    background-color:#ffffff;
    -webkit-box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-tag-input.bp3-active.bp3-intent-primary{
      -webkit-box-shadow:0 0 0 1px #106ba3, 0 0 0 3px rgba(16, 107, 163, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px #106ba3, 0 0 0 3px rgba(16, 107, 163, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-tag-input.bp3-active.bp3-intent-success{
      -webkit-box-shadow:0 0 0 1px #0d8050, 0 0 0 3px rgba(13, 128, 80, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px #0d8050, 0 0 0 3px rgba(13, 128, 80, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-tag-input.bp3-active.bp3-intent-warning{
      -webkit-box-shadow:0 0 0 1px #bf7326, 0 0 0 3px rgba(191, 115, 38, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px #bf7326, 0 0 0 3px rgba(191, 115, 38, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-tag-input.bp3-active.bp3-intent-danger{
      -webkit-box-shadow:0 0 0 1px #c23030, 0 0 0 3px rgba(194, 48, 48, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px #c23030, 0 0 0 3px rgba(194, 48, 48, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
  .bp3-dark .bp3-tag-input .bp3-tag-input-icon, .bp3-tag-input.bp3-dark .bp3-tag-input-icon{
    color:#a7b6c2; }
  .bp3-dark .bp3-tag-input .bp3-input-ghost, .bp3-tag-input.bp3-dark .bp3-input-ghost{
    color:#f5f8fa; }
    .bp3-dark .bp3-tag-input .bp3-input-ghost::-webkit-input-placeholder, .bp3-tag-input.bp3-dark .bp3-input-ghost::-webkit-input-placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-tag-input .bp3-input-ghost::-moz-placeholder, .bp3-tag-input.bp3-dark .bp3-input-ghost::-moz-placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-tag-input .bp3-input-ghost:-ms-input-placeholder, .bp3-tag-input.bp3-dark .bp3-input-ghost:-ms-input-placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-tag-input .bp3-input-ghost::-ms-input-placeholder, .bp3-tag-input.bp3-dark .bp3-input-ghost::-ms-input-placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-tag-input .bp3-input-ghost::placeholder, .bp3-tag-input.bp3-dark .bp3-input-ghost::placeholder{
      color:rgba(167, 182, 194, 0.6); }
  .bp3-dark .bp3-tag-input.bp3-active, .bp3-tag-input.bp3-dark.bp3-active{
    background-color:rgba(16, 22, 26, 0.3);
    -webkit-box-shadow:0 0 0 1px #137cbd, 0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px #137cbd, 0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
    .bp3-dark .bp3-tag-input.bp3-active.bp3-intent-primary, .bp3-tag-input.bp3-dark.bp3-active.bp3-intent-primary{
      -webkit-box-shadow:0 0 0 1px #106ba3, 0 0 0 3px rgba(16, 107, 163, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px #106ba3, 0 0 0 3px rgba(16, 107, 163, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
    .bp3-dark .bp3-tag-input.bp3-active.bp3-intent-success, .bp3-tag-input.bp3-dark.bp3-active.bp3-intent-success{
      -webkit-box-shadow:0 0 0 1px #0d8050, 0 0 0 3px rgba(13, 128, 80, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px #0d8050, 0 0 0 3px rgba(13, 128, 80, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
    .bp3-dark .bp3-tag-input.bp3-active.bp3-intent-warning, .bp3-tag-input.bp3-dark.bp3-active.bp3-intent-warning{
      -webkit-box-shadow:0 0 0 1px #bf7326, 0 0 0 3px rgba(191, 115, 38, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px #bf7326, 0 0 0 3px rgba(191, 115, 38, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
    .bp3-dark .bp3-tag-input.bp3-active.bp3-intent-danger, .bp3-tag-input.bp3-dark.bp3-active.bp3-intent-danger{
      -webkit-box-shadow:0 0 0 1px #c23030, 0 0 0 3px rgba(194, 48, 48, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px #c23030, 0 0 0 3px rgba(194, 48, 48, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }

.bp3-input-ghost{
  background:none;
  border:none;
  -webkit-box-shadow:none;
          box-shadow:none;
  padding:0; }
  .bp3-input-ghost::-webkit-input-placeholder{
    color:rgba(92, 112, 128, 0.6);
    opacity:1; }
  .bp3-input-ghost::-moz-placeholder{
    color:rgba(92, 112, 128, 0.6);
    opacity:1; }
  .bp3-input-ghost:-ms-input-placeholder{
    color:rgba(92, 112, 128, 0.6);
    opacity:1; }
  .bp3-input-ghost::-ms-input-placeholder{
    color:rgba(92, 112, 128, 0.6);
    opacity:1; }
  .bp3-input-ghost::placeholder{
    color:rgba(92, 112, 128, 0.6);
    opacity:1; }
  .bp3-input-ghost:focus{
    outline:none !important; }
.bp3-toast{
  -webkit-box-align:start;
      -ms-flex-align:start;
          align-items:flex-start;
  background-color:#ffffff;
  border-radius:3px;
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 2px 4px rgba(16, 22, 26, 0.2), 0 8px 24px rgba(16, 22, 26, 0.2);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 2px 4px rgba(16, 22, 26, 0.2), 0 8px 24px rgba(16, 22, 26, 0.2);
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  margin:20px 0 0;
  max-width:500px;
  min-width:300px;
  pointer-events:all;
  position:relative !important; }
  .bp3-toast.bp3-toast-enter, .bp3-toast.bp3-toast-appear{
    -webkit-transform:translateY(-40px);
            transform:translateY(-40px); }
  .bp3-toast.bp3-toast-enter-active, .bp3-toast.bp3-toast-appear-active{
    -webkit-transform:translateY(0);
            transform:translateY(0);
    -webkit-transition-delay:0;
            transition-delay:0;
    -webkit-transition-duration:300ms;
            transition-duration:300ms;
    -webkit-transition-property:-webkit-transform;
    transition-property:-webkit-transform;
    transition-property:transform;
    transition-property:transform, -webkit-transform;
    -webkit-transition-timing-function:cubic-bezier(0.54, 1.12, 0.38, 1.11);
            transition-timing-function:cubic-bezier(0.54, 1.12, 0.38, 1.11); }
  .bp3-toast.bp3-toast-enter ~ .bp3-toast, .bp3-toast.bp3-toast-appear ~ .bp3-toast{
    -webkit-transform:translateY(-40px);
            transform:translateY(-40px); }
  .bp3-toast.bp3-toast-enter-active ~ .bp3-toast, .bp3-toast.bp3-toast-appear-active ~ .bp3-toast{
    -webkit-transform:translateY(0);
            transform:translateY(0);
    -webkit-transition-delay:0;
            transition-delay:0;
    -webkit-transition-duration:300ms;
            transition-duration:300ms;
    -webkit-transition-property:-webkit-transform;
    transition-property:-webkit-transform;
    transition-property:transform;
    transition-property:transform, -webkit-transform;
    -webkit-transition-timing-function:cubic-bezier(0.54, 1.12, 0.38, 1.11);
            transition-timing-function:cubic-bezier(0.54, 1.12, 0.38, 1.11); }
  .bp3-toast.bp3-toast-exit{
    opacity:1;
    -webkit-filter:blur(0);
            filter:blur(0); }
  .bp3-toast.bp3-toast-exit-active{
    opacity:0;
    -webkit-filter:blur(10px);
            filter:blur(10px);
    -webkit-transition-delay:0;
            transition-delay:0;
    -webkit-transition-duration:300ms;
            transition-duration:300ms;
    -webkit-transition-property:opacity, -webkit-filter;
    transition-property:opacity, -webkit-filter;
    transition-property:opacity, filter;
    transition-property:opacity, filter, -webkit-filter;
    -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
            transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-toast.bp3-toast-exit ~ .bp3-toast{
    -webkit-transform:translateY(0);
            transform:translateY(0); }
  .bp3-toast.bp3-toast-exit-active ~ .bp3-toast{
    -webkit-transform:translateY(-40px);
            transform:translateY(-40px);
    -webkit-transition-delay:50ms;
            transition-delay:50ms;
    -webkit-transition-duration:100ms;
            transition-duration:100ms;
    -webkit-transition-property:-webkit-transform;
    transition-property:-webkit-transform;
    transition-property:transform;
    transition-property:transform, -webkit-transform;
    -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
            transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-toast .bp3-button-group{
    -webkit-box-flex:0;
        -ms-flex:0 0 auto;
            flex:0 0 auto;
    padding:5px;
    padding-left:0; }
  .bp3-toast > .bp3-icon{
    color:#5c7080;
    margin:12px;
    margin-right:0; }
  .bp3-toast.bp3-dark,
  .bp3-dark .bp3-toast{
    background-color:#394b59;
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 2px 4px rgba(16, 22, 26, 0.4), 0 8px 24px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 2px 4px rgba(16, 22, 26, 0.4), 0 8px 24px rgba(16, 22, 26, 0.4); }
    .bp3-toast.bp3-dark > .bp3-icon,
    .bp3-dark .bp3-toast > .bp3-icon{
      color:#a7b6c2; }
  .bp3-toast[class*="bp3-intent-"] a{
    color:rgba(255, 255, 255, 0.7); }
    .bp3-toast[class*="bp3-intent-"] a:hover{
      color:#ffffff; }
  .bp3-toast[class*="bp3-intent-"] > .bp3-icon{
    color:#ffffff; }
  .bp3-toast[class*="bp3-intent-"] .bp3-button, .bp3-toast[class*="bp3-intent-"] .bp3-button::before,
  .bp3-toast[class*="bp3-intent-"] .bp3-button .bp3-icon, .bp3-toast[class*="bp3-intent-"] .bp3-button:active{
    color:rgba(255, 255, 255, 0.7) !important; }
  .bp3-toast[class*="bp3-intent-"] .bp3-button:focus{
    outline-color:rgba(255, 255, 255, 0.5); }
  .bp3-toast[class*="bp3-intent-"] .bp3-button:hover{
    background-color:rgba(255, 255, 255, 0.15) !important;
    color:#ffffff !important; }
  .bp3-toast[class*="bp3-intent-"] .bp3-button:active{
    background-color:rgba(255, 255, 255, 0.3) !important;
    color:#ffffff !important; }
  .bp3-toast[class*="bp3-intent-"] .bp3-button::after{
    background:rgba(255, 255, 255, 0.3) !important; }
  .bp3-toast.bp3-intent-primary{
    background-color:#137cbd;
    color:#ffffff; }
  .bp3-toast.bp3-intent-success{
    background-color:#0f9960;
    color:#ffffff; }
  .bp3-toast.bp3-intent-warning{
    background-color:#d9822b;
    color:#ffffff; }
  .bp3-toast.bp3-intent-danger{
    background-color:#db3737;
    color:#ffffff; }

.bp3-toast-message{
  -webkit-box-flex:1;
      -ms-flex:1 1 auto;
          flex:1 1 auto;
  padding:11px;
  word-break:break-word; }

.bp3-toast-container{
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  display:-webkit-box !important;
  display:-ms-flexbox !important;
  display:flex !important;
  -webkit-box-orient:vertical;
  -webkit-box-direction:normal;
      -ms-flex-direction:column;
          flex-direction:column;
  left:0;
  overflow:hidden;
  padding:0 20px 20px;
  pointer-events:none;
  right:0;
  z-index:40; }
  .bp3-toast-container.bp3-toast-container-in-portal{
    position:fixed; }
  .bp3-toast-container.bp3-toast-container-inline{
    position:absolute; }
  .bp3-toast-container.bp3-toast-container-top{
    top:0; }
  .bp3-toast-container.bp3-toast-container-bottom{
    bottom:0;
    -webkit-box-orient:vertical;
    -webkit-box-direction:reverse;
        -ms-flex-direction:column-reverse;
            flex-direction:column-reverse;
    top:auto; }
  .bp3-toast-container.bp3-toast-container-left{
    -webkit-box-align:start;
        -ms-flex-align:start;
            align-items:flex-start; }
  .bp3-toast-container.bp3-toast-container-right{
    -webkit-box-align:end;
        -ms-flex-align:end;
            align-items:flex-end; }

.bp3-toast-container-bottom .bp3-toast.bp3-toast-enter:not(.bp3-toast-enter-active),
.bp3-toast-container-bottom .bp3-toast.bp3-toast-enter:not(.bp3-toast-enter-active) ~ .bp3-toast, .bp3-toast-container-bottom .bp3-toast.bp3-toast-appear:not(.bp3-toast-appear-active),
.bp3-toast-container-bottom .bp3-toast.bp3-toast-appear:not(.bp3-toast-appear-active) ~ .bp3-toast,
.bp3-toast-container-bottom .bp3-toast.bp3-toast-exit-active ~ .bp3-toast,
.bp3-toast-container-bottom .bp3-toast.bp3-toast-leave-active ~ .bp3-toast{
  -webkit-transform:translateY(60px);
          transform:translateY(60px); }
.bp3-tooltip{
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 2px 4px rgba(16, 22, 26, 0.2), 0 8px 24px rgba(16, 22, 26, 0.2);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 2px 4px rgba(16, 22, 26, 0.2), 0 8px 24px rgba(16, 22, 26, 0.2);
  -webkit-transform:scale(1);
          transform:scale(1); }
  .bp3-tooltip .bp3-popover-arrow{
    height:22px;
    position:absolute;
    width:22px; }
    .bp3-tooltip .bp3-popover-arrow::before{
      height:14px;
      margin:4px;
      width:14px; }
  .bp3-tether-element-attached-bottom.bp3-tether-target-attached-top > .bp3-tooltip{
    margin-bottom:11px;
    margin-top:-11px; }
    .bp3-tether-element-attached-bottom.bp3-tether-target-attached-top > .bp3-tooltip > .bp3-popover-arrow{
      bottom:-8px; }
      .bp3-tether-element-attached-bottom.bp3-tether-target-attached-top > .bp3-tooltip > .bp3-popover-arrow svg{
        -webkit-transform:rotate(-90deg);
                transform:rotate(-90deg); }
  .bp3-tether-element-attached-left.bp3-tether-target-attached-right > .bp3-tooltip{
    margin-left:11px; }
    .bp3-tether-element-attached-left.bp3-tether-target-attached-right > .bp3-tooltip > .bp3-popover-arrow{
      left:-8px; }
      .bp3-tether-element-attached-left.bp3-tether-target-attached-right > .bp3-tooltip > .bp3-popover-arrow svg{
        -webkit-transform:rotate(0);
                transform:rotate(0); }
  .bp3-tether-element-attached-top.bp3-tether-target-attached-bottom > .bp3-tooltip{
    margin-top:11px; }
    .bp3-tether-element-attached-top.bp3-tether-target-attached-bottom > .bp3-tooltip > .bp3-popover-arrow{
      top:-8px; }
      .bp3-tether-element-attached-top.bp3-tether-target-attached-bottom > .bp3-tooltip > .bp3-popover-arrow svg{
        -webkit-transform:rotate(90deg);
                transform:rotate(90deg); }
  .bp3-tether-element-attached-right.bp3-tether-target-attached-left > .bp3-tooltip{
    margin-left:-11px;
    margin-right:11px; }
    .bp3-tether-element-attached-right.bp3-tether-target-attached-left > .bp3-tooltip > .bp3-popover-arrow{
      right:-8px; }
      .bp3-tether-element-attached-right.bp3-tether-target-attached-left > .bp3-tooltip > .bp3-popover-arrow svg{
        -webkit-transform:rotate(180deg);
                transform:rotate(180deg); }
  .bp3-tether-element-attached-middle > .bp3-tooltip > .bp3-popover-arrow{
    top:50%;
    -webkit-transform:translateY(-50%);
            transform:translateY(-50%); }
  .bp3-tether-element-attached-center > .bp3-tooltip > .bp3-popover-arrow{
    right:50%;
    -webkit-transform:translateX(50%);
            transform:translateX(50%); }
  .bp3-tether-element-attached-top.bp3-tether-target-attached-top > .bp3-tooltip > .bp3-popover-arrow{
    top:-0.22183px; }
  .bp3-tether-element-attached-right.bp3-tether-target-attached-right > .bp3-tooltip > .bp3-popover-arrow{
    right:-0.22183px; }
  .bp3-tether-element-attached-left.bp3-tether-target-attached-left > .bp3-tooltip > .bp3-popover-arrow{
    left:-0.22183px; }
  .bp3-tether-element-attached-bottom.bp3-tether-target-attached-bottom > .bp3-tooltip > .bp3-popover-arrow{
    bottom:-0.22183px; }
  .bp3-tether-element-attached-top.bp3-tether-element-attached-left > .bp3-tooltip{
    -webkit-transform-origin:top left;
            transform-origin:top left; }
  .bp3-tether-element-attached-top.bp3-tether-element-attached-center > .bp3-tooltip{
    -webkit-transform-origin:top center;
            transform-origin:top center; }
  .bp3-tether-element-attached-top.bp3-tether-element-attached-right > .bp3-tooltip{
    -webkit-transform-origin:top right;
            transform-origin:top right; }
  .bp3-tether-element-attached-middle.bp3-tether-element-attached-left > .bp3-tooltip{
    -webkit-transform-origin:center left;
            transform-origin:center left; }
  .bp3-tether-element-attached-middle.bp3-tether-element-attached-center > .bp3-tooltip{
    -webkit-transform-origin:center center;
            transform-origin:center center; }
  .bp3-tether-element-attached-middle.bp3-tether-element-attached-right > .bp3-tooltip{
    -webkit-transform-origin:center right;
            transform-origin:center right; }
  .bp3-tether-element-attached-bottom.bp3-tether-element-attached-left > .bp3-tooltip{
    -webkit-transform-origin:bottom left;
            transform-origin:bottom left; }
  .bp3-tether-element-attached-bottom.bp3-tether-element-attached-center > .bp3-tooltip{
    -webkit-transform-origin:bottom center;
            transform-origin:bottom center; }
  .bp3-tether-element-attached-bottom.bp3-tether-element-attached-right > .bp3-tooltip{
    -webkit-transform-origin:bottom right;
            transform-origin:bottom right; }
  .bp3-tooltip .bp3-popover-content{
    background:#394b59;
    color:#f5f8fa; }
  .bp3-tooltip .bp3-popover-arrow::before{
    -webkit-box-shadow:1px 1px 6px rgba(16, 22, 26, 0.2);
            box-shadow:1px 1px 6px rgba(16, 22, 26, 0.2); }
  .bp3-tooltip .bp3-popover-arrow-border{
    fill:#10161a;
    fill-opacity:0.1; }
  .bp3-tooltip .bp3-popover-arrow-fill{
    fill:#394b59; }
  .bp3-popover-enter > .bp3-tooltip, .bp3-popover-appear > .bp3-tooltip{
    -webkit-transform:scale(0.8);
            transform:scale(0.8); }
  .bp3-popover-enter-active > .bp3-tooltip, .bp3-popover-appear-active > .bp3-tooltip{
    -webkit-transform:scale(1);
            transform:scale(1);
    -webkit-transition-delay:0;
            transition-delay:0;
    -webkit-transition-duration:100ms;
            transition-duration:100ms;
    -webkit-transition-property:-webkit-transform;
    transition-property:-webkit-transform;
    transition-property:transform;
    transition-property:transform, -webkit-transform;
    -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
            transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-popover-exit > .bp3-tooltip{
    -webkit-transform:scale(1);
            transform:scale(1); }
  .bp3-popover-exit-active > .bp3-tooltip{
    -webkit-transform:scale(0.8);
            transform:scale(0.8);
    -webkit-transition-delay:0;
            transition-delay:0;
    -webkit-transition-duration:100ms;
            transition-duration:100ms;
    -webkit-transition-property:-webkit-transform;
    transition-property:-webkit-transform;
    transition-property:transform;
    transition-property:transform, -webkit-transform;
    -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
            transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-tooltip .bp3-popover-content{
    padding:10px 12px; }
  .bp3-tooltip.bp3-dark,
  .bp3-dark .bp3-tooltip{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 2px 4px rgba(16, 22, 26, 0.4), 0 8px 24px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 2px 4px rgba(16, 22, 26, 0.4), 0 8px 24px rgba(16, 22, 26, 0.4); }
    .bp3-tooltip.bp3-dark .bp3-popover-content,
    .bp3-dark .bp3-tooltip .bp3-popover-content{
      background:#e1e8ed;
      color:#394b59; }
    .bp3-tooltip.bp3-dark .bp3-popover-arrow::before,
    .bp3-dark .bp3-tooltip .bp3-popover-arrow::before{
      -webkit-box-shadow:1px 1px 6px rgba(16, 22, 26, 0.4);
              box-shadow:1px 1px 6px rgba(16, 22, 26, 0.4); }
    .bp3-tooltip.bp3-dark .bp3-popover-arrow-border,
    .bp3-dark .bp3-tooltip .bp3-popover-arrow-border{
      fill:#10161a;
      fill-opacity:0.2; }
    .bp3-tooltip.bp3-dark .bp3-popover-arrow-fill,
    .bp3-dark .bp3-tooltip .bp3-popover-arrow-fill{
      fill:#e1e8ed; }
  .bp3-tooltip.bp3-intent-primary .bp3-popover-content{
    background:#137cbd;
    color:#ffffff; }
  .bp3-tooltip.bp3-intent-primary .bp3-popover-arrow-fill{
    fill:#137cbd; }
  .bp3-tooltip.bp3-intent-success .bp3-popover-content{
    background:#0f9960;
    color:#ffffff; }
  .bp3-tooltip.bp3-intent-success .bp3-popover-arrow-fill{
    fill:#0f9960; }
  .bp3-tooltip.bp3-intent-warning .bp3-popover-content{
    background:#d9822b;
    color:#ffffff; }
  .bp3-tooltip.bp3-intent-warning .bp3-popover-arrow-fill{
    fill:#d9822b; }
  .bp3-tooltip.bp3-intent-danger .bp3-popover-content{
    background:#db3737;
    color:#ffffff; }
  .bp3-tooltip.bp3-intent-danger .bp3-popover-arrow-fill{
    fill:#db3737; }

.bp3-tooltip-indicator{
  border-bottom:dotted 1px;
  cursor:help; }
.bp3-tree .bp3-icon, .bp3-tree .bp3-icon-standard, .bp3-tree .bp3-icon-large{
  color:#5c7080; }
  .bp3-tree .bp3-icon.bp3-intent-primary, .bp3-tree .bp3-icon-standard.bp3-intent-primary, .bp3-tree .bp3-icon-large.bp3-intent-primary{
    color:#137cbd; }
  .bp3-tree .bp3-icon.bp3-intent-success, .bp3-tree .bp3-icon-standard.bp3-intent-success, .bp3-tree .bp3-icon-large.bp3-intent-success{
    color:#0f9960; }
  .bp3-tree .bp3-icon.bp3-intent-warning, .bp3-tree .bp3-icon-standard.bp3-intent-warning, .bp3-tree .bp3-icon-large.bp3-intent-warning{
    color:#d9822b; }
  .bp3-tree .bp3-icon.bp3-intent-danger, .bp3-tree .bp3-icon-standard.bp3-intent-danger, .bp3-tree .bp3-icon-large.bp3-intent-danger{
    color:#db3737; }

.bp3-tree-node-list{
  list-style:none;
  margin:0;
  padding-left:0; }

.bp3-tree-root{
  background-color:transparent;
  cursor:default;
  padding-left:0;
  position:relative; }

.bp3-tree-node-content-0{
  padding-left:0px; }

.bp3-tree-node-content-1{
  padding-left:23px; }

.bp3-tree-node-content-2{
  padding-left:46px; }

.bp3-tree-node-content-3{
  padding-left:69px; }

.bp3-tree-node-content-4{
  padding-left:92px; }

.bp3-tree-node-content-5{
  padding-left:115px; }

.bp3-tree-node-content-6{
  padding-left:138px; }

.bp3-tree-node-content-7{
  padding-left:161px; }

.bp3-tree-node-content-8{
  padding-left:184px; }

.bp3-tree-node-content-9{
  padding-left:207px; }

.bp3-tree-node-content-10{
  padding-left:230px; }

.bp3-tree-node-content-11{
  padding-left:253px; }

.bp3-tree-node-content-12{
  padding-left:276px; }

.bp3-tree-node-content-13{
  padding-left:299px; }

.bp3-tree-node-content-14{
  padding-left:322px; }

.bp3-tree-node-content-15{
  padding-left:345px; }

.bp3-tree-node-content-16{
  padding-left:368px; }

.bp3-tree-node-content-17{
  padding-left:391px; }

.bp3-tree-node-content-18{
  padding-left:414px; }

.bp3-tree-node-content-19{
  padding-left:437px; }

.bp3-tree-node-content-20{
  padding-left:460px; }

.bp3-tree-node-content{
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  height:30px;
  padding-right:5px;
  width:100%; }
  .bp3-tree-node-content:hover{
    background-color:rgba(191, 204, 214, 0.4); }

.bp3-tree-node-caret,
.bp3-tree-node-caret-none{
  min-width:30px; }

.bp3-tree-node-caret{
  color:#5c7080;
  cursor:pointer;
  padding:7px;
  -webkit-transform:rotate(0deg);
          transform:rotate(0deg);
  -webkit-transition:-webkit-transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9);
  transition:-webkit-transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9);
  transition:transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9);
  transition:transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9), -webkit-transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-tree-node-caret:hover{
    color:#182026; }
  .bp3-dark .bp3-tree-node-caret{
    color:#a7b6c2; }
    .bp3-dark .bp3-tree-node-caret:hover{
      color:#f5f8fa; }
  .bp3-tree-node-caret.bp3-tree-node-caret-open{
    -webkit-transform:rotate(90deg);
            transform:rotate(90deg); }
  .bp3-tree-node-caret.bp3-icon-standard::before{
    content:""; }

.bp3-tree-node-icon{
  margin-right:7px;
  position:relative; }

.bp3-tree-node-label{
  overflow:hidden;
  text-overflow:ellipsis;
  white-space:nowrap;
  word-wrap:normal;
  -webkit-box-flex:1;
      -ms-flex:1 1 auto;
          flex:1 1 auto;
  position:relative;
  -webkit-user-select:none;
     -moz-user-select:none;
      -ms-user-select:none;
          user-select:none; }
  .bp3-tree-node-label span{
    display:inline; }

.bp3-tree-node-secondary-label{
  padding:0 5px;
  -webkit-user-select:none;
     -moz-user-select:none;
      -ms-user-select:none;
          user-select:none; }
  .bp3-tree-node-secondary-label .bp3-popover-wrapper,
  .bp3-tree-node-secondary-label .bp3-popover-target{
    -webkit-box-align:center;
        -ms-flex-align:center;
            align-items:center;
    display:-webkit-box;
    display:-ms-flexbox;
    display:flex; }

.bp3-tree-node.bp3-disabled .bp3-tree-node-content{
  background-color:inherit;
  color:rgba(92, 112, 128, 0.6);
  cursor:not-allowed; }

.bp3-tree-node.bp3-disabled .bp3-tree-node-caret,
.bp3-tree-node.bp3-disabled .bp3-tree-node-icon{
  color:rgba(92, 112, 128, 0.6);
  cursor:not-allowed; }

.bp3-tree-node.bp3-tree-node-selected > .bp3-tree-node-content{
  background-color:#137cbd; }
  .bp3-tree-node.bp3-tree-node-selected > .bp3-tree-node-content,
  .bp3-tree-node.bp3-tree-node-selected > .bp3-tree-node-content .bp3-icon, .bp3-tree-node.bp3-tree-node-selected > .bp3-tree-node-content .bp3-icon-standard, .bp3-tree-node.bp3-tree-node-selected > .bp3-tree-node-content .bp3-icon-large{
    color:#ffffff; }
  .bp3-tree-node.bp3-tree-node-selected > .bp3-tree-node-content .bp3-tree-node-caret::before{
    color:rgba(255, 255, 255, 0.7); }
  .bp3-tree-node.bp3-tree-node-selected > .bp3-tree-node-content .bp3-tree-node-caret:hover::before{
    color:#ffffff; }

.bp3-dark .bp3-tree-node-content:hover{
  background-color:rgba(92, 112, 128, 0.3); }

.bp3-dark .bp3-tree .bp3-icon, .bp3-dark .bp3-tree .bp3-icon-standard, .bp3-dark .bp3-tree .bp3-icon-large{
  color:#a7b6c2; }
  .bp3-dark .bp3-tree .bp3-icon.bp3-intent-primary, .bp3-dark .bp3-tree .bp3-icon-standard.bp3-intent-primary, .bp3-dark .bp3-tree .bp3-icon-large.bp3-intent-primary{
    color:#137cbd; }
  .bp3-dark .bp3-tree .bp3-icon.bp3-intent-success, .bp3-dark .bp3-tree .bp3-icon-standard.bp3-intent-success, .bp3-dark .bp3-tree .bp3-icon-large.bp3-intent-success{
    color:#0f9960; }
  .bp3-dark .bp3-tree .bp3-icon.bp3-intent-warning, .bp3-dark .bp3-tree .bp3-icon-standard.bp3-intent-warning, .bp3-dark .bp3-tree .bp3-icon-large.bp3-intent-warning{
    color:#d9822b; }
  .bp3-dark .bp3-tree .bp3-icon.bp3-intent-danger, .bp3-dark .bp3-tree .bp3-icon-standard.bp3-intent-danger, .bp3-dark .bp3-tree .bp3-icon-large.bp3-intent-danger{
    color:#db3737; }

.bp3-dark .bp3-tree-node.bp3-tree-node-selected > .bp3-tree-node-content{
  background-color:#137cbd; }
.bp3-omnibar{
  -webkit-filter:blur(0);
          filter:blur(0);
  opacity:1;
  background-color:#ffffff;
  border-radius:3px;
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 4px 8px rgba(16, 22, 26, 0.2), 0 18px 46px 6px rgba(16, 22, 26, 0.2);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 4px 8px rgba(16, 22, 26, 0.2), 0 18px 46px 6px rgba(16, 22, 26, 0.2);
  left:calc(50% - 250px);
  top:20vh;
  width:500px;
  z-index:21; }
  .bp3-omnibar.bp3-overlay-enter, .bp3-omnibar.bp3-overlay-appear{
    -webkit-filter:blur(20px);
            filter:blur(20px);
    opacity:0.2; }
  .bp3-omnibar.bp3-overlay-enter-active, .bp3-omnibar.bp3-overlay-appear-active{
    -webkit-filter:blur(0);
            filter:blur(0);
    opacity:1;
    -webkit-transition-delay:0;
            transition-delay:0;
    -webkit-transition-duration:200ms;
            transition-duration:200ms;
    -webkit-transition-property:opacity, -webkit-filter;
    transition-property:opacity, -webkit-filter;
    transition-property:filter, opacity;
    transition-property:filter, opacity, -webkit-filter;
    -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
            transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-omnibar.bp3-overlay-exit{
    -webkit-filter:blur(0);
            filter:blur(0);
    opacity:1; }
  .bp3-omnibar.bp3-overlay-exit-active{
    -webkit-filter:blur(20px);
            filter:blur(20px);
    opacity:0.2;
    -webkit-transition-delay:0;
            transition-delay:0;
    -webkit-transition-duration:200ms;
            transition-duration:200ms;
    -webkit-transition-property:opacity, -webkit-filter;
    transition-property:opacity, -webkit-filter;
    transition-property:filter, opacity;
    transition-property:filter, opacity, -webkit-filter;
    -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
            transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-omnibar .bp3-input{
    background-color:transparent;
    border-radius:0; }
    .bp3-omnibar .bp3-input, .bp3-omnibar .bp3-input:focus{
      -webkit-box-shadow:none;
              box-shadow:none; }
  .bp3-omnibar .bp3-menu{
    background-color:transparent;
    border-radius:0;
    -webkit-box-shadow:inset 0 1px 0 rgba(16, 22, 26, 0.15);
            box-shadow:inset 0 1px 0 rgba(16, 22, 26, 0.15);
    max-height:calc(60vh - 40px);
    overflow:auto; }
    .bp3-omnibar .bp3-menu:empty{
      display:none; }
  .bp3-dark .bp3-omnibar, .bp3-omnibar.bp3-dark{
    background-color:#30404d;
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 4px 8px rgba(16, 22, 26, 0.4), 0 18px 46px 6px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 4px 8px rgba(16, 22, 26, 0.4), 0 18px 46px 6px rgba(16, 22, 26, 0.4); }

.bp3-omnibar-overlay .bp3-overlay-backdrop{
  background-color:rgba(16, 22, 26, 0.2); }

.bp3-select-popover .bp3-popover-content{
  padding:5px; }

.bp3-select-popover .bp3-input-group{
  margin-bottom:0; }

.bp3-select-popover .bp3-menu{
  max-height:300px;
  max-width:400px;
  overflow:auto;
  padding:0; }
  .bp3-select-popover .bp3-menu:not(:first-child){
    padding-top:5px; }

.bp3-multi-select{
  min-width:150px; }

.bp3-multi-select-popover .bp3-menu{
  max-height:300px;
  max-width:400px;
  overflow:auto; }

.bp3-select-popover .bp3-popover-content{
  padding:5px; }

.bp3-select-popover .bp3-input-group{
  margin-bottom:0; }

.bp3-select-popover .bp3-menu{
  max-height:300px;
  max-width:400px;
  overflow:auto;
  padding:0; }
  .bp3-select-popover .bp3-menu:not(:first-child){
    padding-top:5px; }
/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/* This file was auto-generated by ensureUiComponents() in @jupyterlab/buildutils */

/**
 * (DEPRECATED) Support for consuming icons as CSS background images
 */

/* Icons urls */

:root {
  --jp-icon-add: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTE5IDEzaC02djZoLTJ2LTZINXYtMmg2VjVoMnY2aDZ2MnoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-bug: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIj4KICAgIDxwYXRoIGQ9Ik0yMCA4aC0yLjgxYy0uNDUtLjc4LTEuMDctMS40NS0xLjgyLTEuOTZMMTcgNC40MSAxNS41OSAzbC0yLjE3IDIuMTdDMTIuOTYgNS4wNiAxMi40OSA1IDEyIDVjLS40OSAwLS45Ni4wNi0xLjQxLjE3TDguNDEgMyA3IDQuNDFsMS42MiAxLjYzQzcuODggNi41NSA3LjI2IDcuMjIgNi44MSA4SDR2MmgyLjA5Yy0uMDUuMzMtLjA5LjY2LS4wOSAxdjFINHYyaDJ2MWMwIC4zNC4wNC42Ny4wOSAxSDR2MmgyLjgxYzEuMDQgMS43OSAyLjk3IDMgNS4xOSAzczQuMTUtMS4yMSA1LjE5LTNIMjB2LTJoLTIuMDljLjA1LS4zMy4wOS0uNjYuMDktMXYtMWgydi0yaC0ydi0xYzAtLjM0LS4wNC0uNjctLjA5LTFIMjBWOHptLTYgOGgtNHYtMmg0djJ6bTAtNGgtNHYtMmg0djJ6Ii8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-build: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTYiIHZpZXdCb3g9IjAgMCAyNCAyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTE0LjkgMTcuNDVDMTYuMjUgMTcuNDUgMTcuMzUgMTYuMzUgMTcuMzUgMTVDMTcuMzUgMTMuNjUgMTYuMjUgMTIuNTUgMTQuOSAxMi41NUMxMy41NCAxMi41NSAxMi40NSAxMy42NSAxMi40NSAxNUMxMi40NSAxNi4zNSAxMy41NCAxNy40NSAxNC45IDE3LjQ1Wk0yMC4xIDE1LjY4TDIxLjU4IDE2Ljg0QzIxLjcxIDE2Ljk1IDIxLjc1IDE3LjEzIDIxLjY2IDE3LjI5TDIwLjI2IDE5LjcxQzIwLjE3IDE5Ljg2IDIwIDE5LjkyIDE5LjgzIDE5Ljg2TDE4LjA5IDE5LjE2QzE3LjczIDE5LjQ0IDE3LjMzIDE5LjY3IDE2LjkxIDE5Ljg1TDE2LjY0IDIxLjdDMTYuNjIgMjEuODcgMTYuNDcgMjIgMTYuMyAyMkgxMy41QzEzLjMyIDIyIDEzLjE4IDIxLjg3IDEzLjE1IDIxLjdMMTIuODkgMTkuODVDMTIuNDYgMTkuNjcgMTIuMDcgMTkuNDQgMTEuNzEgMTkuMTZMOS45NjAwMiAxOS44NkM5LjgxMDAyIDE5LjkyIDkuNjIwMDIgMTkuODYgOS41NDAwMiAxOS43MUw4LjE0MDAyIDE3LjI5QzguMDUwMDIgMTcuMTMgOC4wOTAwMiAxNi45NSA4LjIyMDAyIDE2Ljg0TDkuNzAwMDIgMTUuNjhMOS42NTAwMSAxNUw5LjcwMDAyIDE0LjMxTDguMjIwMDIgMTMuMTZDOC4wOTAwMiAxMy4wNSA4LjA1MDAyIDEyLjg2IDguMTQwMDIgMTIuNzFMOS41NDAwMiAxMC4yOUM5LjYyMDAyIDEwLjEzIDkuODEwMDIgMTAuMDcgOS45NjAwMiAxMC4xM0wxMS43MSAxMC44NEMxMi4wNyAxMC41NiAxMi40NiAxMC4zMiAxMi44OSAxMC4xNUwxMy4xNSA4LjI4OTk4QzEzLjE4IDguMTI5OTggMTMuMzIgNy45OTk5OCAxMy41IDcuOTk5OThIMTYuM0MxNi40NyA3Ljk5OTk4IDE2LjYyIDguMTI5OTggMTYuNjQgOC4yODk5OEwxNi45MSAxMC4xNUMxNy4zMyAxMC4zMiAxNy43MyAxMC41NiAxOC4wOSAxMC44NEwxOS44MyAxMC4xM0MyMCAxMC4wNyAyMC4xNyAxMC4xMyAyMC4yNiAxMC4yOUwyMS42NiAxMi43MUMyMS43NSAxMi44NiAyMS43MSAxMy4wNSAyMS41OCAxMy4xNkwyMC4xIDE0LjMxTDIwLjE1IDE1TDIwLjEgMTUuNjhaIi8+CiAgICA8cGF0aCBkPSJNNy4zMjk2NiA3LjQ0NDU0QzguMDgzMSA3LjAwOTU0IDguMzM5MzIgNi4wNTMzMiA3LjkwNDMyIDUuMjk5ODhDNy40NjkzMiA0LjU0NjQzIDYuNTA4MSA0LjI4MTU2IDUuNzU0NjYgNC43MTY1NkM1LjM5MTc2IDQuOTI2MDggNS4xMjY5NSA1LjI3MTE4IDUuMDE4NDkgNS42NzU5NEM0LjkxMDA0IDYuMDgwNzEgNC45NjY4MiA2LjUxMTk4IDUuMTc2MzQgNi44NzQ4OEM1LjYxMTM0IDcuNjI4MzIgNi41NzYyMiA3Ljg3OTU0IDcuMzI5NjYgNy40NDQ1NFpNOS42NTcxOCA0Ljc5NTkzTDEwLjg2NzIgNC45NTE3OUMxMC45NjI4IDQuOTc3NDEgMTEuMDQwMiA1LjA3MTMzIDExLjAzODIgNS4xODc5M0wxMS4wMzg4IDYuOTg4OTNDMTEuMDQ1NSA3LjEwMDU0IDEwLjk2MTYgNy4xOTUxOCAxMC44NTUgNy4yMTA1NEw5LjY2MDAxIDcuMzgwODNMOS4yMzkxNSA4LjEzMTg4TDkuNjY5NjEgOS4yNTc0NUM5LjcwNzI5IDkuMzYyNzEgOS42NjkzNCA5LjQ3Njk5IDkuNTc0MDggOS41MzE5OUw4LjAxNTIzIDEwLjQzMkM3LjkxMTMxIDEwLjQ5MiA3Ljc5MzM3IDEwLjQ2NzcgNy43MjEwNSAxMC4zODI0TDYuOTg3NDggOS40MzE4OEw2LjEwOTMxIDkuNDMwODNMNS4zNDcwNCAxMC4zOTA1QzUuMjg5MDkgMTAuNDcwMiA1LjE3MzgzIDEwLjQ5MDUgNS4wNzE4NyAxMC40MzM5TDMuNTEyNDUgOS41MzI5M0MzLjQxMDQ5IDkuNDc2MzMgMy4zNzY0NyA5LjM1NzQxIDMuNDEwNzUgOS4yNTY3OUwzLjg2MzQ3IDguMTQwOTNMMy42MTc0OSA3Ljc3NDg4TDMuNDIzNDcgNy4zNzg4M0wyLjIzMDc1IDcuMjEyOTdDMi4xMjY0NyA3LjE5MjM1IDIuMDQwNDkgNy4xMDM0MiAyLjA0MjQ1IDYuOTg2ODJMMi4wNDE4NyA1LjE4NTgyQzIuMDQzODMgNS4wNjkyMiAyLjExOTA5IDQuOTc5NTggMi4yMTcwNCA0Ljk2OTIyTDMuNDIwNjUgNC43OTM5M0wzLjg2NzQ5IDQuMDI3ODhMMy40MTEwNSAyLjkxNzMxQzMuMzczMzcgMi44MTIwNCAzLjQxMTMxIDIuNjk3NzYgMy41MTUyMyAyLjYzNzc2TDUuMDc0MDggMS43Mzc3NkM1LjE2OTM0IDEuNjgyNzYgNS4yODcyOSAxLjcwNzA0IDUuMzU5NjEgMS43OTIzMUw2LjExOTE1IDIuNzI3ODhMNi45ODAwMSAyLjczODkzTDcuNzI0OTYgMS43ODkyMkM3Ljc5MTU2IDEuNzA0NTggNy45MTU0OCAxLjY3OTIyIDguMDA4NzkgMS43NDA4Mkw5LjU2ODIxIDIuNjQxODJDOS42NzAxNyAyLjY5ODQyIDkuNzEyODUgMi44MTIzNCA5LjY4NzIzIDIuOTA3OTdMOS4yMTcxOCA0LjAzMzgzTDkuNDYzMTYgNC4zOTk4OEw5LjY1NzE4IDQuNzk1OTNaIi8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-caret-down-empty-thin: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIwIDIwIj4KCTxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSIgc2hhcGUtcmVuZGVyaW5nPSJnZW9tZXRyaWNQcmVjaXNpb24iPgoJCTxwb2x5Z29uIGNsYXNzPSJzdDEiIHBvaW50cz0iOS45LDEzLjYgMy42LDcuNCA0LjQsNi42IDkuOSwxMi4yIDE1LjQsNi43IDE2LjEsNy40ICIvPgoJPC9nPgo8L3N2Zz4K);
  --jp-icon-caret-down-empty: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDE4IDE4Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiIHNoYXBlLXJlbmRlcmluZz0iZ2VvbWV0cmljUHJlY2lzaW9uIj4KICAgIDxwYXRoIGQ9Ik01LjIsNS45TDksOS43bDMuOC0zLjhsMS4yLDEuMmwtNC45LDVsLTQuOS01TDUuMiw1Ljl6Ii8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-caret-down: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDE4IDE4Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiIHNoYXBlLXJlbmRlcmluZz0iZ2VvbWV0cmljUHJlY2lzaW9uIj4KICAgIDxwYXRoIGQ9Ik01LjIsNy41TDksMTEuMmwzLjgtMy44SDUuMnoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-caret-left: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDE4IDE4Ij4KCTxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSIgc2hhcGUtcmVuZGVyaW5nPSJnZW9tZXRyaWNQcmVjaXNpb24iPgoJCTxwYXRoIGQ9Ik0xMC44LDEyLjhMNy4xLDlsMy44LTMuOGwwLDcuNkgxMC44eiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-caret-right: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDE4IDE4Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiIHNoYXBlLXJlbmRlcmluZz0iZ2VvbWV0cmljUHJlY2lzaW9uIj4KICAgIDxwYXRoIGQ9Ik03LjIsNS4yTDEwLjksOWwtMy44LDMuOFY1LjJINy4yeiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-caret-up-empty-thin: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIwIDIwIj4KCTxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSIgc2hhcGUtcmVuZGVyaW5nPSJnZW9tZXRyaWNQcmVjaXNpb24iPgoJCTxwb2x5Z29uIGNsYXNzPSJzdDEiIHBvaW50cz0iMTUuNCwxMy4zIDkuOSw3LjcgNC40LDEzLjIgMy42LDEyLjUgOS45LDYuMyAxNi4xLDEyLjYgIi8+Cgk8L2c+Cjwvc3ZnPgo=);
  --jp-icon-caret-up: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDE4IDE4Ij4KCTxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSIgc2hhcGUtcmVuZGVyaW5nPSJnZW9tZXRyaWNQcmVjaXNpb24iPgoJCTxwYXRoIGQ9Ik01LjIsMTAuNUw5LDYuOGwzLjgsMy44SDUuMnoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-case-sensitive: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIwIDIwIj4KICA8ZyBjbGFzcz0ianAtaWNvbjIiIGZpbGw9IiM0MTQxNDEiPgogICAgPHJlY3QgeD0iMiIgeT0iMiIgd2lkdGg9IjE2IiBoZWlnaHQ9IjE2Ii8+CiAgPC9nPgogIDxnIGNsYXNzPSJqcC1pY29uLWFjY2VudDIiIGZpbGw9IiNGRkYiPgogICAgPHBhdGggZD0iTTcuNiw4aDAuOWwzLjUsOGgtMS4xTDEwLDE0SDZsLTAuOSwySDRMNy42LDh6IE04LDkuMUw2LjQsMTNoMy4yTDgsOS4xeiIvPgogICAgPHBhdGggZD0iTTE2LjYsOS44Yy0wLjIsMC4xLTAuNCwwLjEtMC43LDAuMWMtMC4yLDAtMC40LTAuMS0wLjYtMC4yYy0wLjEtMC4xLTAuMi0wLjQtMC4yLTAuNyBjLTAuMywwLjMtMC42LDAuNS0wLjksMC43Yy0wLjMsMC4xLTAuNywwLjItMS4xLDAuMmMtMC4zLDAtMC41LDAtMC43LTAuMWMtMC4yLTAuMS0wLjQtMC4yLTAuNi0wLjNjLTAuMi0wLjEtMC4zLTAuMy0wLjQtMC41IGMtMC4xLTAuMi0wLjEtMC40LTAuMS0wLjdjMC0wLjMsMC4xLTAuNiwwLjItMC44YzAuMS0wLjIsMC4zLTAuNCwwLjQtMC41QzEyLDcsMTIuMiw2LjksMTIuNSw2LjhjMC4yLTAuMSwwLjUtMC4xLDAuNy0wLjIgYzAuMy0wLjEsMC41LTAuMSwwLjctMC4xYzAuMiwwLDAuNC0wLjEsMC42LTAuMWMwLjIsMCwwLjMtMC4xLDAuNC0wLjJjMC4xLTAuMSwwLjItMC4yLDAuMi0wLjRjMC0xLTEuMS0xLTEuMy0xIGMtMC40LDAtMS40LDAtMS40LDEuMmgtMC45YzAtMC40LDAuMS0wLjcsMC4yLTFjMC4xLTAuMiwwLjMtMC40LDAuNS0wLjZjMC4yLTAuMiwwLjUtMC4zLDAuOC0wLjNDMTMuMyw0LDEzLjYsNCwxMy45LDQgYzAuMywwLDAuNSwwLDAuOCwwLjFjMC4zLDAsMC41LDAuMSwwLjcsMC4yYzAuMiwwLjEsMC40LDAuMywwLjUsMC41QzE2LDUsMTYsNS4yLDE2LDUuNnYyLjljMCwwLjIsMCwwLjQsMCwwLjUgYzAsMC4xLDAuMSwwLjIsMC4zLDAuMmMwLjEsMCwwLjIsMCwwLjMsMFY5Ljh6IE0xNS4yLDYuOWMtMS4yLDAuNi0zLjEsMC4yLTMuMSwxLjRjMCwxLjQsMy4xLDEsMy4xLTAuNVY2Ljl6Ii8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-check: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIj4KICAgIDxwYXRoIGQ9Ik05IDE2LjE3TDQuODMgMTJsLTEuNDIgMS40MUw5IDE5IDIxIDdsLTEuNDEtMS40MXoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-circle-empty: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTEyIDJDNi40NyAyIDIgNi40NyAyIDEyczQuNDcgMTAgMTAgMTAgMTAtNC40NyAxMC0xMFMxNy41MyAyIDEyIDJ6bTAgMThjLTQuNDEgMC04LTMuNTktOC04czMuNTktOCA4LTggOCAzLjU5IDggOC0zLjU5IDgtOCA4eiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-circle: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMTggMTgiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPGNpcmNsZSBjeD0iOSIgY3k9IjkiIHI9IjgiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-clear: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8bWFzayBpZD0iZG9udXRIb2xlIj4KICAgIDxyZWN0IHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgZmlsbD0id2hpdGUiIC8+CiAgICA8Y2lyY2xlIGN4PSIxMiIgY3k9IjEyIiByPSI4IiBmaWxsPSJibGFjayIvPgogIDwvbWFzaz4KCiAgPGcgY2xhc3M9ImpwLWljb24zIiBmaWxsPSIjNjE2MTYxIj4KICAgIDxyZWN0IGhlaWdodD0iMTgiIHdpZHRoPSIyIiB4PSIxMSIgeT0iMyIgdHJhbnNmb3JtPSJyb3RhdGUoMzE1LCAxMiwgMTIpIi8+CiAgICA8Y2lyY2xlIGN4PSIxMiIgY3k9IjEyIiByPSIxMCIgbWFzaz0idXJsKCNkb251dEhvbGUpIi8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-close: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbi1ub25lIGpwLWljb24tc2VsZWN0YWJsZS1pbnZlcnNlIGpwLWljb24zLWhvdmVyIiBmaWxsPSJub25lIj4KICAgIDxjaXJjbGUgY3g9IjEyIiBjeT0iMTIiIHI9IjExIi8+CiAgPC9nPgoKICA8ZyBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIGpwLWljb24tYWNjZW50Mi1ob3ZlciIgZmlsbD0iIzYxNjE2MSI+CiAgICA8cGF0aCBkPSJNMTkgNi40MUwxNy41OSA1IDEyIDEwLjU5IDYuNDEgNSA1IDYuNDEgMTAuNTkgMTIgNSAxNy41OSA2LjQxIDE5IDEyIDEzLjQxIDE3LjU5IDE5IDE5IDE3LjU5IDEzLjQxIDEyeiIvPgogIDwvZz4KCiAgPGcgY2xhc3M9ImpwLWljb24tbm9uZSBqcC1pY29uLWJ1c3kiIGZpbGw9Im5vbmUiPgogICAgPGNpcmNsZSBjeD0iMTIiIGN5PSIxMiIgcj0iNyIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-code: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjIiIGhlaWdodD0iMjIiIHZpZXdCb3g9IjAgMCAyOCAyOCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KCTxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CgkJPHBhdGggZD0iTTExLjQgMTguNkw2LjggMTRMMTEuNCA5LjRMMTAgOEw0IDE0TDEwIDIwTDExLjQgMTguNlpNMTYuNiAxOC42TDIxLjIgMTRMMTYuNiA5LjRMMTggOEwyNCAxNEwxOCAyMEwxNi42IDE4LjZWMTguNloiLz4KCTwvZz4KPC9zdmc+Cg==);
  --jp-icon-console: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIwMCAyMDAiPgogIDxnIGNsYXNzPSJqcC1pY29uLWJyYW5kMSBqcC1pY29uLXNlbGVjdGFibGUiIGZpbGw9IiMwMjg4RDEiPgogICAgPHBhdGggZD0iTTIwIDE5LjhoMTYwdjE1OS45SDIweiIvPgogIDwvZz4KICA8ZyBjbGFzcz0ianAtaWNvbi1zZWxlY3RhYmxlLWludmVyc2UiIGZpbGw9IiNmZmYiPgogICAgPHBhdGggZD0iTTEwNSAxMjcuM2g0MHYxMi44aC00MHpNNTEuMSA3N0w3NCA5OS45bC0yMy4zIDIzLjMgMTAuNSAxMC41IDIzLjMtMjMuM0w5NSA5OS45IDg0LjUgODkuNCA2MS42IDY2LjV6Ii8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-copy: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMTggMTgiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTExLjksMUgzLjJDMi40LDEsMS43LDEuNywxLjcsMi41djEwLjJoMS41VjIuNWg4LjdWMXogTTE0LjEsMy45aC04Yy0wLjgsMC0xLjUsMC43LTEuNSwxLjV2MTAuMmMwLDAuOCwwLjcsMS41LDEuNSwxLjVoOCBjMC44LDAsMS41LTAuNywxLjUtMS41VjUuNEMxNS41LDQuNiwxNC45LDMuOSwxNC4xLDMuOXogTTE0LjEsMTUuNWgtOFY1LjRoOFYxNS41eiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-copyright: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIGVuYWJsZS1iYWNrZ3JvdW5kPSJuZXcgMCAwIDI0IDI0IiBoZWlnaHQ9IjI0IiB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIyNCI+CiAgPGcgY2xhc3M9ImpwLWljb24zIiBmaWxsPSIjNjE2MTYxIj4KICAgIDxwYXRoIGQ9Ik0xMS44OCw5LjE0YzEuMjgsMC4wNiwxLjYxLDEuMTUsMS42MywxLjY2aDEuNzljLTAuMDgtMS45OC0xLjQ5LTMuMTktMy40NS0zLjE5QzkuNjQsNy42MSw4LDksOCwxMi4xNCBjMCwxLjk0LDAuOTMsNC4yNCwzLjg0LDQuMjRjMi4yMiwwLDMuNDEtMS42NSwzLjQ0LTIuOTVoLTEuNzljLTAuMDMsMC41OS0wLjQ1LDEuMzgtMS42MywxLjQ0QzEwLjU1LDE0LjgzLDEwLDEzLjgxLDEwLDEyLjE0IEMxMCw5LjI1LDExLjI4LDkuMTYsMTEuODgsOS4xNHogTTEyLDJDNi40OCwyLDIsNi40OCwyLDEyczQuNDgsMTAsMTAsMTBzMTAtNC40OCwxMC0xMFMxNy41MiwyLDEyLDJ6IE0xMiwyMGMtNC40MSwwLTgtMy41OS04LTggczMuNTktOCw4LThzOCwzLjU5LDgsOFMxNi40MSwyMCwxMiwyMHoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-cut: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTkuNjQgNy42NGMuMjMtLjUuMzYtMS4wNS4zNi0xLjY0IDAtMi4yMS0xLjc5LTQtNC00UzIgMy43OSAyIDZzMS43OSA0IDQgNGMuNTkgMCAxLjE0LS4xMyAxLjY0LS4zNkwxMCAxMmwtMi4zNiAyLjM2QzcuMTQgMTQuMTMgNi41OSAxNCA2IDE0Yy0yLjIxIDAtNCAxLjc5LTQgNHMxLjc5IDQgNCA0IDQtMS43OSA0LTRjMC0uNTktLjEzLTEuMTQtLjM2LTEuNjRMMTIgMTRsNyA3aDN2LTFMOS42NCA3LjY0ek02IDhjLTEuMSAwLTItLjg5LTItMnMuOS0yIDItMiAyIC44OSAyIDItLjkgMi0yIDJ6bTAgMTJjLTEuMSAwLTItLjg5LTItMnMuOS0yIDItMiAyIC44OSAyIDItLjkgMi0yIDJ6bTYtNy41Yy0uMjggMC0uNS0uMjItLjUtLjVzLjIyLS41LjUtLjUuNS4yMi41LjUtLjIyLjUtLjUuNXpNMTkgM2wtNiA2IDIgMiA3LTdWM3oiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-download: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTE5IDloLTRWM0g5djZINWw3IDcgNy03ek01IDE4djJoMTR2LTJINXoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-edit: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTMgMTcuMjVWMjFoMy43NUwxNy44MSA5Ljk0bC0zLjc1LTMuNzVMMyAxNy4yNXpNMjAuNzEgNy4wNGMuMzktLjM5LjM5LTEuMDIgMC0xLjQxbC0yLjM0LTIuMzRjLS4zOS0uMzktMS4wMi0uMzktMS40MSAwbC0xLjgzIDEuODMgMy43NSAzLjc1IDEuODMtMS44M3oiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-ellipses: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPGNpcmNsZSBjeD0iNSIgY3k9IjEyIiByPSIyIi8+CiAgICA8Y2lyY2xlIGN4PSIxMiIgY3k9IjEyIiByPSIyIi8+CiAgICA8Y2lyY2xlIGN4PSIxOSIgY3k9IjEyIiByPSIyIi8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-extension: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTIwLjUgMTFIMTlWN2MwLTEuMS0uOS0yLTItMmgtNFYzLjVDMTMgMi4xMiAxMS44OCAxIDEwLjUgMVM4IDIuMTIgOCAzLjVWNUg0Yy0xLjEgMC0xLjk5LjktMS45OSAydjMuOEgzLjVjMS40OSAwIDIuNyAxLjIxIDIuNyAyLjdzLTEuMjEgMi43LTIuNyAyLjdIMlYyMGMwIDEuMS45IDIgMiAyaDMuOHYtMS41YzAtMS40OSAxLjIxLTIuNyAyLjctMi43IDEuNDkgMCAyLjcgMS4yMSAyLjcgMi43VjIySDE3YzEuMSAwIDItLjkgMi0ydi00aDEuNWMxLjM4IDAgMi41LTEuMTIgMi41LTIuNVMyMS44OCAxMSAyMC41IDExeiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-fast-forward: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGggZD0iTTQgMThsOC41LTZMNCA2djEyem05LTEydjEybDguNS02TDEzIDZ6Ii8+CiAgICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-file-upload: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTkgMTZoNnYtNmg0bC03LTctNyA3aDR6bS00IDJoMTR2Mkg1eiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-file: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8cGF0aCBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIiBkPSJNMTkuMyA4LjJsLTUuNS01LjVjLS4zLS4zLS43LS41LTEuMi0uNUgzLjljLS44LjEtMS42LjktMS42IDEuOHYxNC4xYzAgLjkuNyAxLjYgMS42IDEuNmgxNC4yYy45IDAgMS42LS43IDEuNi0xLjZWOS40Yy4xLS41LS4xLS45LS40LTEuMnptLTUuOC0zLjNsMy40IDMuNmgtMy40VjQuOXptMy45IDEyLjdINC43Yy0uMSAwLS4yIDAtLjItLjJWNC43YzAtLjIuMS0uMy4yLS4zaDcuMnY0LjRzMCAuOC4zIDEuMWMuMy4zIDEuMS4zIDEuMS4zaDQuM3Y3LjJzLS4xLjItLjIuMnoiLz4KPC9zdmc+Cg==);
  --jp-icon-filter-list: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTEwIDE4aDR2LTJoLTR2MnpNMyA2djJoMThWNkgzem0zIDdoMTJ2LTJINnYyeiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-folder: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8cGF0aCBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIiBkPSJNMTAgNEg0Yy0xLjEgMC0xLjk5LjktMS45OSAyTDIgMThjMCAxLjEuOSAyIDIgMmgxNmMxLjEgMCAyLS45IDItMlY4YzAtMS4xLS45LTItMi0yaC04bC0yLTJ6Ii8+Cjwvc3ZnPgo=);
  --jp-icon-html5: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDUxMiA1MTIiPgogIDxwYXRoIGNsYXNzPSJqcC1pY29uMCBqcC1pY29uLXNlbGVjdGFibGUiIGZpbGw9IiMwMDAiIGQ9Ik0xMDguNCAwaDIzdjIyLjhoMjEuMlYwaDIzdjY5aC0yM1Y0NmgtMjF2MjNoLTIzLjJNMjA2IDIzaC0yMC4zVjBoNjMuN3YyM0gyMjl2NDZoLTIzbTUzLjUtNjloMjQuMWwxNC44IDI0LjNMMzEzLjIgMGgyNC4xdjY5aC0yM1YzNC44bC0xNi4xIDI0LjgtMTYuMS0yNC44VjY5aC0yMi42bTg5LjItNjloMjN2NDYuMmgzMi42VjY5aC01NS42Ii8+CiAgPHBhdGggY2xhc3M9ImpwLWljb24tc2VsZWN0YWJsZSIgZmlsbD0iI2U0NGQyNiIgZD0iTTEwNy42IDQ3MWwtMzMtMzcwLjRoMzYyLjhsLTMzIDM3MC4yTDI1NS43IDUxMiIvPgogIDxwYXRoIGNsYXNzPSJqcC1pY29uLXNlbGVjdGFibGUiIGZpbGw9IiNmMTY1MjkiIGQ9Ik0yNTYgNDgwLjVWMTMxaDE0OC4zTDM3NiA0NDciLz4KICA8cGF0aCBjbGFzcz0ianAtaWNvbi1zZWxlY3RhYmxlLWludmVyc2UiIGZpbGw9IiNlYmViZWIiIGQ9Ik0xNDIgMTc2LjNoMTE0djQ1LjRoLTY0LjJsNC4yIDQ2LjVoNjB2NDUuM0gxNTQuNG0yIDIyLjhIMjAybDMuMiAzNi4zIDUwLjggMTMuNnY0Ny40bC05My4yLTI2Ii8+CiAgPHBhdGggY2xhc3M9ImpwLWljb24tc2VsZWN0YWJsZS1pbnZlcnNlIiBmaWxsPSIjZmZmIiBkPSJNMzY5LjYgMTc2LjNIMjU1Ljh2NDUuNGgxMDkuNm0tNC4xIDQ2LjVIMjU1Ljh2NDUuNGg1NmwtNS4zIDU5LTUwLjcgMTMuNnY0Ny4ybDkzLTI1LjgiLz4KPC9zdmc+Cg==);
  --jp-icon-image: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8cGF0aCBjbGFzcz0ianAtaWNvbi1icmFuZDQganAtaWNvbi1zZWxlY3RhYmxlLWludmVyc2UiIGZpbGw9IiNGRkYiIGQ9Ik0yLjIgMi4yaDE3LjV2MTcuNUgyLjJ6Ii8+CiAgPHBhdGggY2xhc3M9ImpwLWljb24tYnJhbmQwIGpwLWljb24tc2VsZWN0YWJsZSIgZmlsbD0iIzNGNTFCNSIgZD0iTTIuMiAyLjJ2MTcuNWgxNy41bC4xLTE3LjVIMi4yem0xMi4xIDIuMmMxLjIgMCAyLjIgMSAyLjIgMi4ycy0xIDIuMi0yLjIgMi4yLTIuMi0xLTIuMi0yLjIgMS0yLjIgMi4yLTIuMnpNNC40IDE3LjZsMy4zLTguOCAzLjMgNi42IDIuMi0zLjIgNC40IDUuNEg0LjR6Ii8+Cjwvc3ZnPgo=);
  --jp-icon-inspector: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8cGF0aCBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIiBkPSJNMjAgNEg0Yy0xLjEgMC0xLjk5LjktMS45OSAyTDIgMThjMCAxLjEuOSAyIDIgMmgxNmMxLjEgMCAyLS45IDItMlY2YzAtMS4xLS45LTItMi0yem0tNSAxNEg0di00aDExdjR6bTAtNUg0VjloMTF2NHptNSA1aC00VjloNHY5eiIvPgo8L3N2Zz4K);
  --jp-icon-json: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8ZyBjbGFzcz0ianAtaWNvbi13YXJuMSBqcC1pY29uLXNlbGVjdGFibGUiIGZpbGw9IiNGOUE4MjUiPgogICAgPHBhdGggZD0iTTIwLjIgMTEuOGMtMS42IDAtMS43LjUtMS43IDEgMCAuNC4xLjkuMSAxLjMuMS41LjEuOS4xIDEuMyAwIDEuNy0xLjQgMi4zLTMuNSAyLjNoLS45di0xLjloLjVjMS4xIDAgMS40IDAgMS40LS44IDAtLjMgMC0uNi0uMS0xIDAtLjQtLjEtLjgtLjEtMS4yIDAtMS4zIDAtMS44IDEuMy0yLTEuMy0uMi0xLjMtLjctMS4zLTIgMC0uNC4xLS44LjEtMS4yLjEtLjQuMS0uNy4xLTEgMC0uOC0uNC0uNy0xLjQtLjhoLS41VjQuMWguOWMyLjIgMCAzLjUuNyAzLjUgMi4zIDAgLjQtLjEuOS0uMSAxLjMtLjEuNS0uMS45LS4xIDEuMyAwIC41LjIgMSAxLjcgMXYxLjh6TTEuOCAxMC4xYzEuNiAwIDEuNy0uNSAxLjctMSAwLS40LS4xLS45LS4xLTEuMy0uMS0uNS0uMS0uOS0uMS0xLjMgMC0xLjYgMS40LTIuMyAzLjUtMi4zaC45djEuOWgtLjVjLTEgMC0xLjQgMC0xLjQuOCAwIC4zIDAgLjYuMSAxIDAgLjIuMS42LjEgMSAwIDEuMyAwIDEuOC0xLjMgMkM2IDExLjIgNiAxMS43IDYgMTNjMCAuNC0uMS44LS4xIDEuMi0uMS4zLS4xLjctLjEgMSAwIC44LjMuOCAxLjQuOGguNXYxLjloLS45Yy0yLjEgMC0zLjUtLjYtMy41LTIuMyAwLS40LjEtLjkuMS0xLjMuMS0uNS4xLS45LjEtMS4zIDAtLjUtLjItMS0xLjctMXYtMS45eiIvPgogICAgPGNpcmNsZSBjeD0iMTEiIGN5PSIxMy44IiByPSIyLjEiLz4KICAgIDxjaXJjbGUgY3g9IjExIiBjeT0iOC4yIiByPSIyLjEiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-julia: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDMyNSAzMDAiPgogIDxnIGNsYXNzPSJqcC1icmFuZDAganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjY2IzYzMzIj4KICAgIDxwYXRoIGQ9Ik0gMTUwLjg5ODQzOCAyMjUgQyAxNTAuODk4NDM4IDI2Ni40MjE4NzUgMTE3LjMyMDMxMiAzMDAgNzUuODk4NDM4IDMwMCBDIDM0LjQ3NjU2MiAzMDAgMC44OTg0MzggMjY2LjQyMTg3NSAwLjg5ODQzOCAyMjUgQyAwLjg5ODQzOCAxODMuNTc4MTI1IDM0LjQ3NjU2MiAxNTAgNzUuODk4NDM4IDE1MCBDIDExNy4zMjAzMTIgMTUwIDE1MC44OTg0MzggMTgzLjU3ODEyNSAxNTAuODk4NDM4IDIyNSIvPgogIDwvZz4KICA8ZyBjbGFzcz0ianAtYnJhbmQwIGpwLWljb24tc2VsZWN0YWJsZSIgZmlsbD0iIzM4OTgyNiI+CiAgICA8cGF0aCBkPSJNIDIzNy41IDc1IEMgMjM3LjUgMTE2LjQyMTg3NSAyMDMuOTIxODc1IDE1MCAxNjIuNSAxNTAgQyAxMjEuMDc4MTI1IDE1MCA4Ny41IDExNi40MjE4NzUgODcuNSA3NSBDIDg3LjUgMzMuNTc4MTI1IDEyMS4wNzgxMjUgMCAxNjIuNSAwIEMgMjAzLjkyMTg3NSAwIDIzNy41IDMzLjU3ODEyNSAyMzcuNSA3NSIvPgogIDwvZz4KICA8ZyBjbGFzcz0ianAtYnJhbmQwIGpwLWljb24tc2VsZWN0YWJsZSIgZmlsbD0iIzk1NThiMiI+CiAgICA8cGF0aCBkPSJNIDMyNC4xMDE1NjIgMjI1IEMgMzI0LjEwMTU2MiAyNjYuNDIxODc1IDI5MC41MjM0MzggMzAwIDI0OS4xMDE1NjIgMzAwIEMgMjA3LjY3OTY4OCAzMDAgMTc0LjEwMTU2MiAyNjYuNDIxODc1IDE3NC4xMDE1NjIgMjI1IEMgMTc0LjEwMTU2MiAxODMuNTc4MTI1IDIwNy42Nzk2ODggMTUwIDI0OS4xMDE1NjIgMTUwIEMgMjkwLjUyMzQzOCAxNTAgMzI0LjEwMTU2MiAxODMuNTc4MTI1IDMyNC4xMDE1NjIgMjI1Ii8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-jupyter-favicon: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTUyIiBoZWlnaHQ9IjE2NSIgdmlld0JveD0iMCAwIDE1MiAxNjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbi13YXJuMCIgZmlsbD0iI0YzNzcyNiI+CiAgICA8cGF0aCB0cmFuc2Zvcm09InRyYW5zbGF0ZSgwLjA3ODk0NywgMTEwLjU4MjkyNykiIGQ9Ik03NS45NDIyODQyLDI5LjU4MDQ1NjEgQzQzLjMwMjM5NDcsMjkuNTgwNDU2MSAxNC43OTY3ODMyLDE3LjY1MzQ2MzQgMCwwIEM1LjUxMDgzMjExLDE1Ljg0MDY4MjkgMTUuNzgxNTM4OSwyOS41NjY3NzMyIDI5LjM5MDQ5NDcsMzkuMjc4NDE3MSBDNDIuOTk5Nyw0OC45ODk4NTM3IDU5LjI3MzcsNTQuMjA2NzgwNSA3NS45NjA1Nzg5LDU0LjIwNjc4MDUgQzkyLjY0NzQ1NzksNTQuMjA2NzgwNSAxMDguOTIxNDU4LDQ4Ljk4OTg1MzcgMTIyLjUzMDY2MywzOS4yNzg0MTcxIEMxMzYuMTM5NDUzLDI5LjU2Njc3MzIgMTQ2LjQxMDI4NCwxNS44NDA2ODI5IDE1MS45MjExNTgsMCBDMTM3LjA4Nzg2OCwxNy42NTM0NjM0IDEwOC41ODI1ODksMjkuNTgwNDU2MSA3NS45NDIyODQyLDI5LjU4MDQ1NjEgTDc1Ljk0MjI4NDIsMjkuNTgwNDU2MSBaIiAvPgogICAgPHBhdGggdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMC4wMzczNjgsIDAuNzA0ODc4KSIgZD0iTTc1Ljk3ODQ1NzksMjQuNjI2NDA3MyBDMTA4LjYxODc2MywyNC42MjY0MDczIDEzNy4xMjQ0NTgsMzYuNTUzNDQxNSAxNTEuOTIxMTU4LDU0LjIwNjc4MDUgQzE0Ni40MTAyODQsMzguMzY2MjIyIDEzNi4xMzk0NTMsMjQuNjQwMTMxNyAxMjIuNTMwNjYzLDE0LjkyODQ4NzggQzEwOC45MjE0NTgsNS4yMTY4NDM5IDkyLjY0NzQ1NzksMCA3NS45NjA1Nzg5LDAgQzU5LjI3MzcsMCA0Mi45OTk3LDUuMjE2ODQzOSAyOS4zOTA0OTQ3LDE0LjkyODQ4NzggQzE1Ljc4MTUzODksMjQuNjQwMTMxNyA1LjUxMDgzMjExLDM4LjM2NjIyMiAwLDU0LjIwNjc4MDUgQzE0LjgzMzA4MTYsMzYuNTg5OTI5MyA0My4zMzg1Njg0LDI0LjYyNjQwNzMgNzUuOTc4NDU3OSwyNC42MjY0MDczIEw3NS45Nzg0NTc5LDI0LjYyNjQwNzMgWiIgLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-jupyter: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzkiIGhlaWdodD0iNTEiIHZpZXdCb3g9IjAgMCAzOSA1MSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSgtMTYzOCAtMjI4MSkiPgogICAgPGcgY2xhc3M9ImpwLWljb24td2FybjAiIGZpbGw9IiNGMzc3MjYiPgogICAgICA8cGF0aCB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxNjM5Ljc0IDIzMTEuOTgpIiBkPSJNIDE4LjI2NDYgNy4xMzQxMUMgMTAuNDE0NSA3LjEzNDExIDMuNTU4NzIgNC4yNTc2IDAgMEMgMS4zMjUzOSAzLjgyMDQgMy43OTU1NiA3LjEzMDgxIDcuMDY4NiA5LjQ3MzAzQyAxMC4zNDE3IDExLjgxNTIgMTQuMjU1NyAxMy4wNzM0IDE4LjI2OSAxMy4wNzM0QyAyMi4yODIzIDEzLjA3MzQgMjYuMTk2MyAxMS44MTUyIDI5LjQ2OTQgOS40NzMwM0MgMzIuNzQyNCA3LjEzMDgxIDM1LjIxMjYgMy44MjA0IDM2LjUzOCAwQyAzMi45NzA1IDQuMjU3NiAyNi4xMTQ4IDcuMTM0MTEgMTguMjY0NiA3LjEzNDExWiIvPgogICAgICA8cGF0aCB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxNjM5LjczIDIyODUuNDgpIiBkPSJNIDE4LjI3MzMgNS45MzkzMUMgMjYuMTIzNSA1LjkzOTMxIDMyLjk3OTMgOC44MTU4MyAzNi41MzggMTMuMDczNEMgMzUuMjEyNiA5LjI1MzAzIDMyLjc0MjQgNS45NDI2MiAyOS40Njk0IDMuNjAwNEMgMjYuMTk2MyAxLjI1ODE4IDIyLjI4MjMgMCAxOC4yNjkgMEMgMTQuMjU1NyAwIDEwLjM0MTcgMS4yNTgxOCA3LjA2ODYgMy42MDA0QyAzLjc5NTU2IDUuOTQyNjIgMS4zMjUzOSA5LjI1MzAzIDAgMTMuMDczNEMgMy41Njc0NSA4LjgyNDYzIDEwLjQyMzIgNS45MzkzMSAxOC4yNzMzIDUuOTM5MzFaIi8+CiAgICA8L2c+CiAgICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgICA8cGF0aCB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxNjY5LjMgMjI4MS4zMSkiIGQ9Ik0gNS44OTM1MyAyLjg0NEMgNS45MTg4OSAzLjQzMTY1IDUuNzcwODUgNC4wMTM2NyA1LjQ2ODE1IDQuNTE2NDVDIDUuMTY1NDUgNS4wMTkyMiA0LjcyMTY4IDUuNDIwMTUgNC4xOTI5OSA1LjY2ODUxQyAzLjY2NDMgNS45MTY4OCAzLjA3NDQ0IDYuMDAxNTEgMi40OTgwNSA1LjkxMTcxQyAxLjkyMTY2IDUuODIxOSAxLjM4NDYzIDUuNTYxNyAwLjk1NDg5OCA1LjE2NDAxQyAwLjUyNTE3IDQuNzY2MzMgMC4yMjIwNTYgNC4yNDkwMyAwLjA4MzkwMzcgMy42Nzc1N0MgLTAuMDU0MjQ4MyAzLjEwNjExIC0wLjAyMTIzIDIuNTA2MTcgMC4xNzg3ODEgMS45NTM2NEMgMC4zNzg3OTMgMS40MDExIDAuNzM2ODA5IDAuOTIwODE3IDEuMjA3NTQgMC41NzM1MzhDIDEuNjc4MjYgMC4yMjYyNTkgMi4yNDA1NSAwLjAyNzU5MTkgMi44MjMyNiAwLjAwMjY3MjI5QyAzLjYwMzg5IC0wLjAzMDcxMTUgNC4zNjU3MyAwLjI0OTc4OSA0Ljk0MTQyIDAuNzgyNTUxQyA1LjUxNzExIDEuMzE1MzEgNS44NTk1NiAyLjA1Njc2IDUuODkzNTMgMi44NDRaIi8+CiAgICAgIDxwYXRoIHRyYW5zZm9ybT0idHJhbnNsYXRlKDE2MzkuOCAyMzIzLjgxKSIgZD0iTSA3LjQyNzg5IDMuNTgzMzhDIDcuNDYwMDggNC4zMjQzIDcuMjczNTUgNS4wNTgxOSA2Ljg5MTkzIDUuNjkyMTNDIDYuNTEwMzEgNi4zMjYwNyA1Ljk1MDc1IDYuODMxNTYgNS4yODQxMSA3LjE0NDZDIDQuNjE3NDcgNy40NTc2MyAzLjg3MzcxIDcuNTY0MTQgMy4xNDcwMiA3LjQ1MDYzQyAyLjQyMDMyIDcuMzM3MTIgMS43NDMzNiA3LjAwODcgMS4yMDE4NCA2LjUwNjk1QyAwLjY2MDMyOCA2LjAwNTIgMC4yNzg2MSA1LjM1MjY4IDAuMTA1MDE3IDQuNjMyMDJDIC0wLjA2ODU3NTcgMy45MTEzNSAtMC4wMjYyMzYxIDMuMTU0OTQgMC4yMjY2NzUgMi40NTg1NkMgMC40Nzk1ODcgMS43NjIxNyAwLjkzMTY5NyAxLjE1NzEzIDEuNTI1NzYgMC43MjAwMzNDIDIuMTE5ODMgMC4yODI5MzUgMi44MjkxNCAwLjAzMzQzOTUgMy41NjM4OSAwLjAwMzEzMzQ0QyA0LjU0NjY3IC0wLjAzNzQwMzMgNS41MDUyOSAwLjMxNjcwNiA2LjIyOTYxIDAuOTg3ODM1QyA2Ljk1MzkzIDEuNjU4OTYgNy4zODQ4NCAyLjU5MjM1IDcuNDI3ODkgMy41ODMzOEwgNy40Mjc4OSAzLjU4MzM4WiIvPgogICAgICA8cGF0aCB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxNjM4LjM2IDIyODYuMDYpIiBkPSJNIDIuMjc0NzEgNC4zOTYyOUMgMS44NDM2MyA0LjQxNTA4IDEuNDE2NzEgNC4zMDQ0NSAxLjA0Nzk5IDQuMDc4NDNDIDAuNjc5MjY4IDMuODUyNCAwLjM4NTMyOCAzLjUyMTE0IDAuMjAzMzcxIDMuMTI2NTZDIDAuMDIxNDEzNiAyLjczMTk4IC0wLjA0MDM3OTggMi4yOTE4MyAwLjAyNTgxMTYgMS44NjE4MUMgMC4wOTIwMDMxIDEuNDMxOCAwLjI4MzIwNCAxLjAzMTI2IDAuNTc1MjEzIDAuNzEwODgzQyAwLjg2NzIyMiAwLjM5MDUxIDEuMjQ2OTEgMC4xNjQ3MDggMS42NjYyMiAwLjA2MjA1OTJDIDIuMDg1NTMgLTAuMDQwNTg5NyAyLjUyNTYxIC0wLjAxNTQ3MTQgMi45MzA3NiAwLjEzNDIzNUMgMy4zMzU5MSAwLjI4Mzk0MSAzLjY4NzkyIDAuNTUxNTA1IDMuOTQyMjIgMC45MDMwNkMgNC4xOTY1MiAxLjI1NDYyIDQuMzQxNjkgMS42NzQzNiA0LjM1OTM1IDIuMTA5MTZDIDQuMzgyOTkgMi42OTEwNyA0LjE3Njc4IDMuMjU4NjkgMy43ODU5NyAzLjY4NzQ2QyAzLjM5NTE2IDQuMTE2MjQgMi44NTE2NiA0LjM3MTE2IDIuMjc0NzEgNC4zOTYyOUwgMi4yNzQ3MSA0LjM5NjI5WiIvPgogICAgPC9nPgogIDwvZz4+Cjwvc3ZnPgo=);
  --jp-icon-jupyterlab-wordmark: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyMDAiIHZpZXdCb3g9IjAgMCAxODYwLjggNDc1Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjIiIGZpbGw9IiM0RTRFNEUiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDQ4MC4xMzY0MDEsIDY0LjI3MTQ5MykiPgogICAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMC4wMDAwMDAsIDU4Ljg3NTU2NikiPgogICAgICA8ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSgwLjA4NzYwMywgMC4xNDAyOTQpIj4KICAgICAgICA8cGF0aCBkPSJNLTQyNi45LDE2OS44YzAsNDguNy0zLjcsNjQuNy0xMy42LDc2LjRjLTEwLjgsMTAtMjUsMTUuNS0zOS43LDE1LjVsMy43LDI5IGMyMi44LDAuMyw0NC44LTcuOSw2MS45LTIzLjFjMTcuOC0xOC41LDI0LTQ0LjEsMjQtODMuM1YwSC00Mjd2MTcwLjFMLTQyNi45LDE2OS44TC00MjYuOSwxNjkuOHoiLz4KICAgICAgPC9nPgogICAgPC9nPgogICAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTU1LjA0NTI5NiwgNTYuODM3MTA0KSI+CiAgICAgIDxnIHRyYW5zZm9ybT0idHJhbnNsYXRlKDEuNTYyNDUzLCAxLjc5OTg0MikiPgogICAgICAgIDxwYXRoIGQ9Ik0tMzEyLDE0OGMwLDIxLDAsMzkuNSwxLjcsNTUuNGgtMzEuOGwtMi4xLTMzLjNoLTAuOGMtNi43LDExLjYtMTYuNCwyMS4zLTI4LDI3LjkgYy0xMS42LDYuNi0yNC44LDEwLTM4LjIsOS44Yy0zMS40LDAtNjktMTcuNy02OS04OVYwaDM2LjR2MTEyLjdjMCwzOC43LDExLjYsNjQuNyw0NC42LDY0LjdjMTAuMy0wLjIsMjAuNC0zLjUsMjguOS05LjQgYzguNS01LjksMTUuMS0xNC4zLDE4LjktMjMuOWMyLjItNi4xLDMuMy0xMi41LDMuMy0xOC45VjAuMmgzNi40VjE0OEgtMzEyTC0zMTIsMTQ4eiIvPgogICAgICA8L2c+CiAgICA8L2c+CiAgICA8ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSgzOTAuMDEzMzIyLCA1My40Nzk2MzgpIj4KICAgICAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMS43MDY0NTgsIDAuMjMxNDI1KSI+CiAgICAgICAgPHBhdGggZD0iTS00NzguNiw3MS40YzAtMjYtMC44LTQ3LTEuNy02Ni43aDMyLjdsMS43LDM0LjhoMC44YzcuMS0xMi41LDE3LjUtMjIuOCwzMC4xLTI5LjcgYzEyLjUtNywyNi43LTEwLjMsNDEtOS44YzQ4LjMsMCw4NC43LDQxLjcsODQuNywxMDMuM2MwLDczLjEtNDMuNywxMDkuMi05MSwxMDkuMmMtMTIuMSwwLjUtMjQuMi0yLjItMzUtNy44IGMtMTAuOC01LjYtMTkuOS0xMy45LTI2LjYtMjQuMmgtMC44VjI5MWgtMzZ2LTIyMEwtNDc4LjYsNzEuNEwtNDc4LjYsNzEuNHogTS00NDIuNiwxMjUuNmMwLjEsNS4xLDAuNiwxMC4xLDEuNywxNS4xIGMzLDEyLjMsOS45LDIzLjMsMTkuOCwzMS4xYzkuOSw3LjgsMjIuMSwxMi4xLDM0LjcsMTIuMWMzOC41LDAsNjAuNy0zMS45LDYwLjctNzguNWMwLTQwLjctMjEuMS03NS42LTU5LjUtNzUuNiBjLTEyLjksMC40LTI1LjMsNS4xLTM1LjMsMTMuNGMtOS45LDguMy0xNi45LDE5LjctMTkuNiwzMi40Yy0xLjUsNC45LTIuMywxMC0yLjUsMTUuMVYxMjUuNkwtNDQyLjYsMTI1LjZMLTQ0Mi42LDEyNS42eiIvPgogICAgICA8L2c+CiAgICA8L2c+CiAgICA8ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSg2MDYuNzQwNzI2LCA1Ni44MzcxMDQpIj4KICAgICAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMC43NTEyMjYsIDEuOTg5Mjk5KSI+CiAgICAgICAgPHBhdGggZD0iTS00NDAuOCwwbDQzLjcsMTIwLjFjNC41LDEzLjQsOS41LDI5LjQsMTIuOCw0MS43aDAuOGMzLjctMTIuMiw3LjktMjcuNywxMi44LTQyLjQgbDM5LjctMTE5LjJoMzguNUwtMzQ2LjksMTQ1Yy0yNiw2OS43LTQzLjcsMTA1LjQtNjguNiwxMjcuMmMtMTIuNSwxMS43LTI3LjksMjAtNDQuNiwyMy45bC05LjEtMzEuMSBjMTEuNy0zLjksMjIuNS0xMC4xLDMxLjgtMTguMWMxMy4yLTExLjEsMjMuNy0yNS4yLDMwLjYtNDEuMmMxLjUtMi44LDIuNS01LjcsMi45LTguOGMtMC4zLTMuMy0xLjItNi42LTIuNS05LjdMLTQ4MC4yLDAuMSBoMzkuN0wtNDQwLjgsMEwtNDQwLjgsMHoiLz4KICAgICAgPC9nPgogICAgPC9nPgogICAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoODIyLjc0ODEwNCwgMC4wMDAwMDApIj4KICAgICAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMS40NjQwNTAsIDAuMzc4OTE0KSI+CiAgICAgICAgPHBhdGggZD0iTS00MTMuNywwdjU4LjNoNTJ2MjguMmgtNTJWMTk2YzAsMjUsNywzOS41LDI3LjMsMzkuNWM3LjEsMC4xLDE0LjItMC43LDIxLjEtMi41IGwxLjcsMjcuN2MtMTAuMywzLjctMjEuMyw1LjQtMzIuMiw1Yy03LjMsMC40LTE0LjYtMC43LTIxLjMtMy40Yy02LjgtMi43LTEyLjktNi44LTE3LjktMTIuMWMtMTAuMy0xMC45LTE0LjEtMjktMTQuMS01Mi45IFY4Ni41aC0zMVY1OC4zaDMxVjkuNkwtNDEzLjcsMEwtNDEzLjcsMHoiLz4KICAgICAgPC9nPgogICAgPC9nPgogICAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoOTc0LjQzMzI4NiwgNTMuNDc5NjM4KSI+CiAgICAgIDxnIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuOTkwMDM0LCAwLjYxMDMzOSkiPgogICAgICAgIDxwYXRoIGQ9Ik0tNDQ1LjgsMTEzYzAuOCw1MCwzMi4yLDcwLjYsNjguNiw3MC42YzE5LDAuNiwzNy45LTMsNTUuMy0xMC41bDYuMiwyNi40IGMtMjAuOSw4LjktNDMuNSwxMy4xLTY2LjIsMTIuNmMtNjEuNSwwLTk4LjMtNDEuMi05OC4zLTEwMi41Qy00ODAuMiw0OC4yLTQ0NC43LDAtMzg2LjUsMGM2NS4yLDAsODIuNyw1OC4zLDgyLjcsOTUuNyBjLTAuMSw1LjgtMC41LDExLjUtMS4yLDE3LjJoLTE0MC42SC00NDUuOEwtNDQ1LjgsMTEzeiBNLTMzOS4yLDg2LjZjMC40LTIzLjUtOS41LTYwLjEtNTAuNC02MC4xIGMtMzYuOCwwLTUyLjgsMzQuNC01NS43LDYwLjFILTMzOS4yTC0zMzkuMiw4Ni42TC0zMzkuMiw4Ni42eiIvPgogICAgICA8L2c+CiAgICA8L2c+CiAgICA8ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjAxLjk2MTA1OCwgNTMuNDc5NjM4KSI+CiAgICAgIDxnIHRyYW5zZm9ybT0idHJhbnNsYXRlKDEuMTc5NjQwLCAwLjcwNTA2OCkiPgogICAgICAgIDxwYXRoIGQ9Ik0tNDc4LjYsNjhjMC0yMy45LTAuNC00NC41LTEuNy02My40aDMxLjhsMS4yLDM5LjloMS43YzkuMS0yNy4zLDMxLTQ0LjUsNTUuMy00NC41IGMzLjUtMC4xLDcsMC40LDEwLjMsMS4ydjM0LjhjLTQuMS0wLjktOC4yLTEuMy0xMi40LTEuMmMtMjUuNiwwLTQzLjcsMTkuNy00OC43LDQ3LjRjLTEsNS43LTEuNiwxMS41LTEuNywxNy4ydjEwOC4zaC0zNlY2OCBMLTQ3OC42LDY4eiIvPgogICAgICA8L2c+CiAgICA8L2c+CiAgPC9nPgoKICA8ZyBjbGFzcz0ianAtaWNvbi13YXJuMCIgZmlsbD0iI0YzNzcyNiI+CiAgICA8cGF0aCBkPSJNMTM1Mi4zLDMyNi4yaDM3VjI4aC0zN1YzMjYuMnogTTE2MDQuOCwzMjYuMmMtMi41LTEzLjktMy40LTMxLjEtMy40LTQ4Ljd2LTc2IGMwLTQwLjctMTUuMS04My4xLTc3LjMtODMuMWMtMjUuNiwwLTUwLDcuMS02Ni44LDE4LjFsOC40LDI0LjRjMTQuMy05LjIsMzQtMTUuMSw1My0xNS4xYzQxLjYsMCw0Ni4yLDMwLjIsNDYuMiw0N3Y0LjIgYy03OC42LTAuNC0xMjIuMywyNi41LTEyMi4zLDc1LjZjMCwyOS40LDIxLDU4LjQsNjIuMiw1OC40YzI5LDAsNTAuOS0xNC4zLDYyLjItMzAuMmgxLjNsMi45LDI1LjZIMTYwNC44eiBNMTU2NS43LDI1Ny43IGMwLDMuOC0wLjgsOC0yLjEsMTEuOGMtNS45LDE3LjItMjIuNywzNC00OS4yLDM0Yy0xOC45LDAtMzQuOS0xMS4zLTM0LjktMzUuM2MwLTM5LjUsNDUuOC00Ni42LDg2LjItNDUuOFYyNTcuN3ogTTE2OTguNSwzMjYuMiBsMS43LTMzLjZoMS4zYzE1LjEsMjYuOSwzOC43LDM4LjIsNjguMSwzOC4yYzQ1LjQsMCw5MS4yLTM2LjEsOTEuMi0xMDguOGMwLjQtNjEuNy0zNS4zLTEwMy43LTg1LjctMTAzLjcgYy0zMi44LDAtNTYuMywxNC43LTY5LjMsMzcuNGgtMC44VjI4aC0zNi42djI0NS43YzAsMTguMS0wLjgsMzguNi0xLjcsNTIuNUgxNjk4LjV6IE0xNzA0LjgsMjA4LjJjMC01LjksMS4zLTEwLjksMi4xLTE1LjEgYzcuNi0yOC4xLDMxLjEtNDUuNCw1Ni4zLTQ1LjRjMzkuNSwwLDYwLjUsMzQuOSw2MC41LDc1LjZjMCw0Ni42LTIzLjEsNzguMS02MS44LDc4LjFjLTI2LjksMC00OC4zLTE3LjYtNTUuNS00My4zIGMtMC44LTQuMi0xLjctOC44LTEuNy0xMy40VjIwOC4yeiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-kernel: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICAgIDxwYXRoIGNsYXNzPSJqcC1pY29uMiIgZmlsbD0iIzYxNjE2MSIgZD0iTTE1IDlIOXY2aDZWOXptLTIgNGgtMnYtMmgydjJ6bTgtMlY5aC0yVjdjMC0xLjEtLjktMi0yLTJoLTJWM2gtMnYyaC0yVjNIOXYySDdjLTEuMSAwLTIgLjktMiAydjJIM3YyaDJ2MkgzdjJoMnYyYzAgMS4xLjkgMiAyIDJoMnYyaDJ2LTJoMnYyaDJ2LTJoMmMxLjEgMCAyLS45IDItMnYtMmgydi0yaC0ydi0yaDJ6bS00IDZIN1Y3aDEwdjEweiIvPgo8L3N2Zz4K);
  --jp-icon-keyboard: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8cGF0aCBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIiBkPSJNMjAgNUg0Yy0xLjEgMC0xLjk5LjktMS45OSAyTDIgMTdjMCAxLjEuOSAyIDIgMmgxNmMxLjEgMCAyLS45IDItMlY3YzAtMS4xLS45LTItMi0yem0tOSAzaDJ2MmgtMlY4em0wIDNoMnYyaC0ydi0yek04IDhoMnYySDhWOHptMCAzaDJ2Mkg4di0yem0tMSAySDV2LTJoMnYyem0wLTNINVY4aDJ2MnptOSA3SDh2LTJoOHYyem0wLTRoLTJ2LTJoMnYyem0wLTNoLTJWOGgydjJ6bTMgM2gtMnYtMmgydjJ6bTAtM2gtMlY4aDJ2MnoiLz4KPC9zdmc+Cg==);
  --jp-icon-launcher: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8cGF0aCBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIiBkPSJNMTkgMTlINVY1aDdWM0g1YTIgMiAwIDAwLTIgMnYxNGEyIDIgMCAwMDIgMmgxNGMxLjEgMCAyLS45IDItMnYtN2gtMnY3ek0xNCAzdjJoMy41OWwtOS44MyA5LjgzIDEuNDEgMS40MUwxOSA2LjQxVjEwaDJWM2gtN3oiLz4KPC9zdmc+Cg==);
  --jp-icon-line-form: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICAgIDxwYXRoIGZpbGw9IndoaXRlIiBkPSJNNS44OCA0LjEyTDEzLjc2IDEybC03Ljg4IDcuODhMOCAyMmwxMC0xMEw4IDJ6Ii8+Cjwvc3ZnPgo=);
  --jp-icon-link: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTMuOSAxMmMwLTEuNzEgMS4zOS0zLjEgMy4xLTMuMWg0VjdIN2MtMi43NiAwLTUgMi4yNC01IDVzMi4yNCA1IDUgNWg0di0xLjlIN2MtMS43MSAwLTMuMS0xLjM5LTMuMS0zLjF6TTggMTNoOHYtMkg4djJ6bTktNmgtNHYxLjloNGMxLjcxIDAgMy4xIDEuMzkgMy4xIDMuMXMtMS4zOSAzLjEtMy4xIDMuMWgtNFYxN2g0YzIuNzYgMCA1LTIuMjQgNS01cy0yLjI0LTUtNS01eiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-list: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICAgIDxwYXRoIGNsYXNzPSJqcC1pY29uMiBqcC1pY29uLXNlbGVjdGFibGUiIGZpbGw9IiM2MTYxNjEiIGQ9Ik0xOSA1djE0SDVWNWgxNG0xLjEtMkgzLjljLS41IDAtLjkuNC0uOS45djE2LjJjMCAuNC40LjkuOS45aDE2LjJjLjQgMCAuOS0uNS45LS45VjMuOWMwLS41LS41LS45LS45LS45ek0xMSA3aDZ2MmgtNlY3em0wIDRoNnYyaC02di0yem0wIDRoNnYyaC02ek03IDdoMnYySDd6bTAgNGgydjJIN3ptMCA0aDJ2Mkg3eiIvPgo8L3N2Zz4=);
  --jp-icon-listings-info: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA1MC45NzggNTAuOTc4IiBzdHlsZT0iZW5hYmxlLWJhY2tncm91bmQ6bmV3IDAgMCA1MC45NzggNTAuOTc4OyIgeG1sOnNwYWNlPSJwcmVzZXJ2ZSI+Cgk8Zz4KCQk8cGF0aCBzdHlsZT0iZmlsbDojMDEwMDAyOyIgZD0iTTQzLjUyLDcuNDU4QzM4LjcxMSwyLjY0OCwzMi4zMDcsMCwyNS40ODksMEMxOC42NywwLDEyLjI2NiwyLjY0OCw3LjQ1OCw3LjQ1OAoJCQljLTkuOTQzLDkuOTQxLTkuOTQzLDI2LjExOSwwLDM2LjA2MmM0LjgwOSw0LjgwOSwxMS4yMTIsNy40NTYsMTguMDMxLDcuNDU4YzAsMCwwLjAwMSwwLDAuMDAyLDAKCQkJYzYuODE2LDAsMTMuMjIxLTIuNjQ4LDE4LjAyOS03LjQ1OGM0LjgwOS00LjgwOSw3LjQ1Ny0xMS4yMTIsNy40NTctMTguMDNDNTAuOTc3LDE4LjY3LDQ4LjMyOCwxMi4yNjYsNDMuNTIsNy40NTh6CgkJCSBNNDIuMTA2LDQyLjEwNWMtNC40MzIsNC40MzEtMTAuMzMyLDYuODcyLTE2LjYxNSw2Ljg3MmgtMC4wMDJjLTYuMjg1LTAuMDAxLTEyLjE4Ny0yLjQ0MS0xNi42MTctNi44NzIKCQkJYy05LjE2Mi05LjE2My05LjE2Mi0yNC4wNzEsMC0zMy4yMzNDMTMuMzAzLDQuNDQsMTkuMjA0LDIsMjUuNDg5LDJjNi4yODQsMCwxMi4xODYsMi40NCwxNi42MTcsNi44NzIKCQkJYzQuNDMxLDQuNDMxLDYuODcxLDEwLjMzMiw2Ljg3MSwxNi42MTdDNDguOTc3LDMxLjc3Miw0Ni41MzYsMzcuNjc1LDQyLjEwNiw0Mi4xMDV6Ii8+CgkJPHBhdGggc3R5bGU9ImZpbGw6IzAxMDAwMjsiIGQ9Ik0yMy41NzgsMzIuMjE4Yy0wLjAyMy0xLjczNCwwLjE0My0zLjA1OSwwLjQ5Ni0zLjk3MmMwLjM1My0wLjkxMywxLjExLTEuOTk3LDIuMjcyLTMuMjUzCgkJCWMwLjQ2OC0wLjUzNiwwLjkyMy0xLjA2MiwxLjM2Ny0xLjU3NWMwLjYyNi0wLjc1MywxLjEwNC0xLjQ3OCwxLjQzNi0yLjE3NWMwLjMzMS0wLjcwNywwLjQ5NS0xLjU0MSwwLjQ5NS0yLjUKCQkJYzAtMS4wOTYtMC4yNi0yLjA4OC0wLjc3OS0yLjk3OWMtMC41NjUtMC44NzktMS41MDEtMS4zMzYtMi44MDYtMS4zNjljLTEuODAyLDAuMDU3LTIuOTg1LDAuNjY3LTMuNTUsMS44MzIKCQkJYy0wLjMwMSwwLjUzNS0wLjUwMywxLjE0MS0wLjYwNywxLjgxNGMtMC4xMzksMC43MDctMC4yMDcsMS40MzItMC4yMDcsMi4xNzRoLTIuOTM3Yy0wLjA5MS0yLjIwOCwwLjQwNy00LjExNCwxLjQ5My01LjcxOQoJCQljMS4wNjItMS42NCwyLjg1NS0yLjQ4MSw1LjM3OC0yLjUyN2MyLjE2LDAuMDIzLDMuODc0LDAuNjA4LDUuMTQxLDEuNzU4YzEuMjc4LDEuMTYsMS45MjksMi43NjQsMS45NSw0LjgxMQoJCQljMCwxLjE0Mi0wLjEzNywyLjExMS0wLjQxLDIuOTExYy0wLjMwOSwwLjg0NS0wLjczMSwxLjU5My0xLjI2OCwyLjI0M2MtMC40OTIsMC42NS0xLjA2OCwxLjMxOC0xLjczLDIuMDAyCgkJCWMtMC42NSwwLjY5Ny0xLjMxMywxLjQ3OS0xLjk4NywyLjM0NmMtMC4yMzksMC4zNzctMC40MjksMC43NzctMC41NjUsMS4xOTljLTAuMTYsMC45NTktMC4yMTcsMS45NTEtMC4xNzEsMi45NzkKCQkJQzI2LjU4OSwzMi4yMTgsMjMuNTc4LDMyLjIxOCwyMy41NzgsMzIuMjE4eiBNMjMuNTc4LDM4LjIydi0zLjQ4NGgzLjA3NnYzLjQ4NEgyMy41Nzh6Ii8+Cgk8L2c+Cjwvc3ZnPgo=);
  --jp-icon-markdown: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8cGF0aCBjbGFzcz0ianAtaWNvbi1jb250cmFzdDAganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjN0IxRkEyIiBkPSJNNSAxNC45aDEybC02LjEgNnptOS40LTYuOGMwLTEuMy0uMS0yLjktLjEtNC41LS40IDEuNC0uOSAyLjktMS4zIDQuM2wtMS4zIDQuM2gtMkw4LjUgNy45Yy0uNC0xLjMtLjctMi45LTEtNC4zLS4xIDEuNi0uMSAzLjItLjIgNC42TDcgMTIuNEg0LjhsLjctMTFoMy4zTDEwIDVjLjQgMS4yLjcgMi43IDEgMy45LjMtMS4yLjctMi42IDEtMy45bDEuMi0zLjdoMy4zbC42IDExaC0yLjRsLS4zLTQuMnoiLz4KPC9zdmc+Cg==);
  --jp-icon-new-folder: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTIwIDZoLThsLTItMkg0Yy0xLjExIDAtMS45OS44OS0xLjk5IDJMMiAxOGMwIDEuMTEuODkgMiAyIDJoMTZjMS4xMSAwIDItLjg5IDItMlY4YzAtMS4xMS0uODktMi0yLTJ6bS0xIDhoLTN2M2gtMnYtM2gtM3YtMmgzVjloMnYzaDN2MnoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-not-trusted: url(data:image/svg+xml;base64,PHN2ZyBmaWxsPSJub25lIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI1IDI1Ij4KICAgIDxwYXRoIGNsYXNzPSJqcC1pY29uMiIgc3Ryb2tlPSIjMzMzMzMzIiBzdHJva2Utd2lkdGg9IjIiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDMgMykiIGQ9Ik0xLjg2MDk0IDExLjQ0MDlDMC44MjY0NDggOC43NzAyNyAwLjg2Mzc3OSA2LjA1NzY0IDEuMjQ5MDcgNC4xOTkzMkMyLjQ4MjA2IDMuOTMzNDcgNC4wODA2OCAzLjQwMzQ3IDUuNjAxMDIgMi44NDQ5QzcuMjM1NDkgMi4yNDQ0IDguODU2NjYgMS41ODE1IDkuOTg3NiAxLjA5NTM5QzExLjA1OTcgMS41ODM0MSAxMi42MDk0IDIuMjQ0NCAxNC4yMTggMi44NDMzOUMxNS43NTAzIDMuNDEzOTQgMTcuMzk5NSAzLjk1MjU4IDE4Ljc1MzkgNC4yMTM4NUMxOS4xMzY0IDYuMDcxNzcgMTkuMTcwOSA4Ljc3NzIyIDE4LjEzOSAxMS40NDA5QzE3LjAzMDMgMTQuMzAzMiAxNC42NjY4IDE3LjE4NDQgOS45OTk5OSAxOC45MzU0QzUuMzMzMTkgMTcuMTg0NCAyLjk2OTY4IDE0LjMwMzIgMS44NjA5NCAxMS40NDA5WiIvPgogICAgPHBhdGggY2xhc3M9ImpwLWljb24yIiBzdHJva2U9IiMzMzMzMzMiIHN0cm9rZS13aWR0aD0iMiIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoOS4zMTU5MiA5LjMyMDMxKSIgZD0iTTcuMzY4NDIgMEwwIDcuMzY0NzkiLz4KICAgIDxwYXRoIGNsYXNzPSJqcC1pY29uMiIgc3Ryb2tlPSIjMzMzMzMzIiBzdHJva2Utd2lkdGg9IjIiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDkuMzE1OTIgMTYuNjgzNikgc2NhbGUoMSAtMSkiIGQ9Ik03LjM2ODQyIDBMMCA3LjM2NDc5Ii8+Cjwvc3ZnPgo=);
  --jp-icon-notebook: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8ZyBjbGFzcz0ianAtaWNvbi13YXJuMCBqcC1pY29uLXNlbGVjdGFibGUiIGZpbGw9IiNFRjZDMDAiPgogICAgPHBhdGggZD0iTTE4LjcgMy4zdjE1LjRIMy4zVjMuM2gxNS40bTEuNS0xLjVIMS44djE4LjNoMTguM2wuMS0xOC4zeiIvPgogICAgPHBhdGggZD0iTTE2LjUgMTYuNWwtNS40LTQuMy01LjYgNC4zdi0xMWgxMXoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-numbering: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjIiIGhlaWdodD0iMjIiIHZpZXdCb3g9IjAgMCAyOCAyOCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KCTxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CgkJPHBhdGggZD0iTTQgMTlINlYxOS41SDVWMjAuNUg2VjIxSDRWMjJIN1YxOEg0VjE5Wk01IDEwSDZWNkg0VjdINVYxMFpNNCAxM0g1LjhMNCAxNS4xVjE2SDdWMTVINS4yTDcgMTIuOVYxMkg0VjEzWk05IDdWOUgyM1Y3SDlaTTkgMjFIMjNWMTlIOVYyMVpNOSAxNUgyM1YxM0g5VjE1WiIvPgoJPC9nPgo8L3N2Zz4K);
  --jp-icon-offline-bolt: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgd2lkdGg9IjE2Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTEyIDIuMDJjLTUuNTEgMC05Ljk4IDQuNDctOS45OCA5Ljk4czQuNDcgOS45OCA5Ljk4IDkuOTggOS45OC00LjQ3IDkuOTgtOS45OFMxNy41MSAyLjAyIDEyIDIuMDJ6TTExLjQ4IDIwdi02LjI2SDhMMTMgNHY2LjI2aDMuMzVMMTEuNDggMjB6Ii8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-palette: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTE4IDEzVjIwSDRWNkg5LjAyQzkuMDcgNS4yOSA5LjI0IDQuNjIgOS41IDRINEMyLjkgNCAyIDQuOSAyIDZWMjBDMiAyMS4xIDIuOSAyMiA0IDIySDE4QzE5LjEgMjIgMjAgMjEuMSAyMCAyMFYxNUwxOCAxM1pNMTkuMyA4Ljg5QzE5Ljc0IDguMTkgMjAgNy4zOCAyMCA2LjVDMjAgNC4wMSAxNy45OSAyIDE1LjUgMkMxMy4wMSAyIDExIDQuMDEgMTEgNi41QzExIDguOTkgMTMuMDEgMTEgMTUuNDkgMTFDMTYuMzcgMTEgMTcuMTkgMTAuNzQgMTcuODggMTAuM0wyMSAxMy40MkwyMi40MiAxMkwxOS4zIDguODlaTTE1LjUgOUMxNC4xMiA5IDEzIDcuODggMTMgNi41QzEzIDUuMTIgMTQuMTIgNCAxNS41IDRDMTYuODggNCAxOCA1LjEyIDE4IDYuNUMxOCA3Ljg4IDE2Ljg4IDkgMTUuNSA5WiIvPgogICAgPHBhdGggZmlsbC1ydWxlPSJldmVub2RkIiBjbGlwLXJ1bGU9ImV2ZW5vZGQiIGQ9Ik00IDZIOS4wMTg5NEM5LjAwNjM5IDYuMTY1MDIgOSA2LjMzMTc2IDkgNi41QzkgOC44MTU3NyAxMC4yMTEgMTAuODQ4NyAxMi4wMzQzIDEySDlWMTRIMTZWMTIuOTgxMUMxNi41NzAzIDEyLjkzNzcgMTcuMTIgMTIuODIwNyAxNy42Mzk2IDEyLjYzOTZMMTggMTNWMjBINFY2Wk04IDhINlYxMEg4VjhaTTYgMTJIOFYxNEg2VjEyWk04IDE2SDZWMThIOFYxNlpNOSAxNkgxNlYxOEg5VjE2WiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-paste: url(data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjI0IiB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGggZD0iTTE5IDJoLTQuMThDMTQuNC44NCAxMy4zIDAgMTIgMGMtMS4zIDAtMi40Ljg0LTIuODIgMkg1Yy0xLjEgMC0yIC45LTIgMnYxNmMwIDEuMS45IDIgMiAyaDE0YzEuMSAwIDItLjkgMi0yVjRjMC0xLjEtLjktMi0yLTJ6bS03IDBjLjU1IDAgMSAuNDUgMSAxcy0uNDUgMS0xIDEtMS0uNDUtMS0xIC40NS0xIDEtMXptNyAxOEg1VjRoMnYzaDEwVjRoMnYxNnoiLz4KICAgIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-pdf: url(data:image/svg+xml;base64,PHN2ZwogICB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyMiAyMiIgd2lkdGg9IjE2Ij4KICAgIDxwYXRoIHRyYW5zZm9ybT0icm90YXRlKDQ1KSIgY2xhc3M9ImpwLWljb24tc2VsZWN0YWJsZSIgZmlsbD0iI0ZGMkEyQSIKICAgICAgIGQ9Im0gMjIuMzQ0MzY5LC0zLjAxNjM2NDIgaCA1LjYzODYwNCB2IDEuNTc5MjQzMyBoIC0zLjU0OTIyNyB2IDEuNTA4NjkyOTkgaCAzLjMzNzU3NiBWIDEuNjUwODE1NCBoIC0zLjMzNzU3NiB2IDMuNDM1MjYxMyBoIC0yLjA4OTM3NyB6IG0gLTcuMTM2NDQ0LDEuNTc5MjQzMyB2IDQuOTQzOTU0MyBoIDAuNzQ4OTIgcSAxLjI4MDc2MSwwIDEuOTUzNzAzLC0wLjYzNDk1MzUgMC42NzgzNjksLTAuNjM0OTUzNSAwLjY3ODM2OSwtMS44NDUxNjQxIDAsLTEuMjA0NzgzNTUgLTAuNjcyOTQyLC0xLjgzNDMxMDExIC0wLjY3Mjk0MiwtMC42Mjk1MjY1OSAtMS45NTkxMywtMC42Mjk1MjY1OSB6IG0gLTIuMDg5Mzc3LC0xLjU3OTI0MzMgaCAyLjIwMzM0MyBxIDEuODQ1MTY0LDAgMi43NDYwMzksMC4yNjU5MjA3IDAuOTA2MzAxLDAuMjYwNDkzNyAxLjU1MjEwOCwwLjg5MDAyMDMgMC41Njk4MywwLjU0ODEyMjMgMC44NDY2MDUsMS4yNjQ0ODAwNiAwLjI3Njc3NCwwLjcxNjM1NzgxIDAuMjc2Nzc0LDEuNjIyNjU4OTQgMCwwLjkxNzE1NTEgLTAuMjc2Nzc0LDEuNjM4OTM5OSAtMC4yNzY3NzUsMC43MTYzNTc4IC0wLjg0NjYwNSwxLjI2NDQ4IC0wLjY1MTIzNCwwLjYyOTUyNjYgLTEuNTYyOTYyLDAuODk1NDQ3MyAtMC45MTE3MjgsMC4yNjA0OTM3IC0yLjczNTE4NSwwLjI2MDQ5MzcgaCAtMi4yMDMzNDMgeiBtIC04LjE0NTg1NjUsMCBoIDMuNDY3ODIzIHEgMS41NDY2ODE2LDAgMi4zNzE1Nzg1LDAuNjg5MjIzIDAuODMwMzI0LDAuNjgzNzk2MSAwLjgzMDMyNCwxLjk1MzcwMzE0IDAsMS4yNzUzMzM5NyAtMC44MzAzMjQsMS45NjQ1NTcwNiBRIDkuOTg3MTk2MSwyLjI3NDkxNSA4LjQ0MDUxNDUsMi4yNzQ5MTUgSCA3LjA2MjA2ODQgViA1LjA4NjA3NjcgSCA0Ljk3MjY5MTUgWiBtIDIuMDg5Mzc2OSwxLjUxNDExOTkgdiAyLjI2MzAzOTQzIGggMS4xNTU5NDEgcSAwLjYwNzgxODgsMCAwLjkzODg2MjksLTAuMjkzMDU1NDcgMC4zMzEwNDQxLC0wLjI5ODQ4MjQxIDAuMzMxMDQ0MSwtMC44NDExNzc3MiAwLC0wLjU0MjY5NTMxIC0wLjMzMTA0NDEsLTAuODM1NzUwNzQgLTAuMzMxMDQ0MSwtMC4yOTMwNTU1IC0wLjkzODg2MjksLTAuMjkzMDU1NSB6IgovPgo8L3N2Zz4K);
  --jp-icon-python: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8ZyBjbGFzcz0ianAtaWNvbi1icmFuZDAganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjMEQ0N0ExIj4KICAgIDxwYXRoIGQ9Ik0xMS4xIDYuOVY1LjhINi45YzAtLjUgMC0xLjMuMi0xLjYuNC0uNy44LTEuMSAxLjctMS40IDEuNy0uMyAyLjUtLjMgMy45LS4xIDEgLjEgMS45LjkgMS45IDEuOXY0LjJjMCAuNS0uOSAxLjYtMiAxLjZIOC44Yy0xLjUgMC0yLjQgMS40LTIuNCAyLjh2Mi4ySDQuN0MzLjUgMTUuMSAzIDE0IDMgMTMuMVY5Yy0uMS0xIC42LTIgMS44LTIgMS41LS4xIDYuMy0uMSA2LjMtLjF6Ii8+CiAgICA8cGF0aCBkPSJNMTAuOSAxNS4xdjEuMWg0LjJjMCAuNSAwIDEuMy0uMiAxLjYtLjQuNy0uOCAxLjEtMS43IDEuNC0xLjcuMy0yLjUuMy0zLjkuMS0xLS4xLTEuOS0uOS0xLjktMS45di00LjJjMC0uNS45LTEuNiAyLTEuNmgzLjhjMS41IDAgMi40LTEuNCAyLjQtMi44VjYuNmgxLjdDMTguNSA2LjkgMTkgOCAxOSA4LjlWMTNjMCAxLS43IDIuMS0xLjkgMi4xaC02LjJ6Ii8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-r-kernel: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8cGF0aCBjbGFzcz0ianAtaWNvbi1jb250cmFzdDMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjMjE5NkYzIiBkPSJNNC40IDIuNWMxLjItLjEgMi45LS4zIDQuOS0uMyAyLjUgMCA0LjEuNCA1LjIgMS4zIDEgLjcgMS41IDEuOSAxLjUgMy41IDAgMi0xLjQgMy41LTIuOSA0LjEgMS4yLjQgMS43IDEuNiAyLjIgMyAuNiAxLjkgMSAzLjkgMS4zIDQuNmgtMy44Yy0uMy0uNC0uOC0xLjctMS4yLTMuN3MtMS4yLTIuNi0yLjYtMi42aC0uOXY2LjRINC40VjIuNXptMy43IDYuOWgxLjRjMS45IDAgMi45LS45IDIuOS0yLjNzLTEtMi4zLTIuOC0yLjNjLS43IDAtMS4zIDAtMS42LjJ2NC41aC4xdi0uMXoiLz4KPC9zdmc+Cg==);
  --jp-icon-react: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMTUwIDE1MCA1NDEuOSAyOTUuMyI+CiAgPGcgY2xhc3M9ImpwLWljb24tYnJhbmQyIGpwLWljb24tc2VsZWN0YWJsZSIgZmlsbD0iIzYxREFGQiI+CiAgICA8cGF0aCBkPSJNNjY2LjMgMjk2LjVjMC0zMi41LTQwLjctNjMuMy0xMDMuMS04Mi40IDE0LjQtNjMuNiA4LTExNC4yLTIwLjItMTMwLjQtNi41LTMuOC0xNC4xLTUuNi0yMi40LTUuNnYyMi4zYzQuNiAwIDguMy45IDExLjQgMi42IDEzLjYgNy44IDE5LjUgMzcuNSAxNC45IDc1LjctMS4xIDkuNC0yLjkgMTkuMy01LjEgMjkuNC0xOS42LTQuOC00MS04LjUtNjMuNS0xMC45LTEzLjUtMTguNS0yNy41LTM1LjMtNDEuNi01MCAzMi42LTMwLjMgNjMuMi00Ni45IDg0LTQ2LjlWNzhjLTI3LjUgMC02My41IDE5LjYtOTkuOSA1My42LTM2LjQtMzMuOC03Mi40LTUzLjItOTkuOS01My4ydjIyLjNjMjAuNyAwIDUxLjQgMTYuNSA4NCA0Ni42LTE0IDE0LjctMjggMzEuNC00MS4zIDQ5LjktMjIuNiAyLjQtNDQgNi4xLTYzLjYgMTEtMi4zLTEwLTQtMTkuNy01LjItMjktNC43LTM4LjIgMS4xLTY3LjkgMTQuNi03NS44IDMtMS44IDYuOS0yLjYgMTEuNS0yLjZWNzguNWMtOC40IDAtMTYgMS44LTIyLjYgNS42LTI4LjEgMTYuMi0zNC40IDY2LjctMTkuOSAxMzAuMS02Mi4yIDE5LjItMTAyLjcgNDkuOS0xMDIuNyA4Mi4zIDAgMzIuNSA0MC43IDYzLjMgMTAzLjEgODIuNC0xNC40IDYzLjYtOCAxMTQuMiAyMC4yIDEzMC40IDYuNSAzLjggMTQuMSA1LjYgMjIuNSA1LjYgMjcuNSAwIDYzLjUtMTkuNiA5OS45LTUzLjYgMzYuNCAzMy44IDcyLjQgNTMuMiA5OS45IDUzLjIgOC40IDAgMTYtMS44IDIyLjYtNS42IDI4LjEtMTYuMiAzNC40LTY2LjcgMTkuOS0xMzAuMSA2Mi0xOS4xIDEwMi41LTQ5LjkgMTAyLjUtODIuM3ptLTEzMC4yLTY2LjdjLTMuNyAxMi45LTguMyAyNi4yLTEzLjUgMzkuNS00LjEtOC04LjQtMTYtMTMuMS0yNC00LjYtOC05LjUtMTUuOC0xNC40LTIzLjQgMTQuMiAyLjEgMjcuOSA0LjcgNDEgNy45em0tNDUuOCAxMDYuNWMtNy44IDEzLjUtMTUuOCAyNi4zLTI0LjEgMzguMi0xNC45IDEuMy0zMCAyLTQ1LjIgMi0xNS4xIDAtMzAuMi0uNy00NS0xLjktOC4zLTExLjktMTYuNC0yNC42LTI0LjItMzgtNy42LTEzLjEtMTQuNS0yNi40LTIwLjgtMzkuOCA2LjItMTMuNCAxMy4yLTI2LjggMjAuNy0zOS45IDcuOC0xMy41IDE1LjgtMjYuMyAyNC4xLTM4LjIgMTQuOS0xLjMgMzAtMiA0NS4yLTIgMTUuMSAwIDMwLjIuNyA0NSAxLjkgOC4zIDExLjkgMTYuNCAyNC42IDI0LjIgMzggNy42IDEzLjEgMTQuNSAyNi40IDIwLjggMzkuOC02LjMgMTMuNC0xMy4yIDI2LjgtMjAuNyAzOS45em0zMi4zLTEzYzUuNCAxMy40IDEwIDI2LjggMTMuOCAzOS44LTEzLjEgMy4yLTI2LjkgNS45LTQxLjIgOCA0LjktNy43IDkuOC0xNS42IDE0LjQtMjMuNyA0LjYtOCA4LjktMTYuMSAxMy0yNC4xek00MjEuMiA0MzBjLTkuMy05LjYtMTguNi0yMC4zLTI3LjgtMzIgOSAuNCAxOC4yLjcgMjcuNS43IDkuNCAwIDE4LjctLjIgMjcuOC0uNy05IDExLjctMTguMyAyMi40LTI3LjUgMzJ6bS03NC40LTU4LjljLTE0LjItMi4xLTI3LjktNC43LTQxLTcuOSAzLjctMTIuOSA4LjMtMjYuMiAxMy41LTM5LjUgNC4xIDggOC40IDE2IDEzLjEgMjQgNC43IDggOS41IDE1LjggMTQuNCAyMy40ek00MjAuNyAxNjNjOS4zIDkuNiAxOC42IDIwLjMgMjcuOCAzMi05LS40LTE4LjItLjctMjcuNS0uNy05LjQgMC0xOC43LjItMjcuOC43IDktMTEuNyAxOC4zLTIyLjQgMjcuNS0zMnptLTc0IDU4LjljLTQuOSA3LjctOS44IDE1LjYtMTQuNCAyMy43LTQuNiA4LTguOSAxNi0xMyAyNC01LjQtMTMuNC0xMC0yNi44LTEzLjgtMzkuOCAxMy4xLTMuMSAyNi45LTUuOCA0MS4yLTcuOXptLTkwLjUgMTI1LjJjLTM1LjQtMTUuMS01OC4zLTM0LjktNTguMy01MC42IDAtMTUuNyAyMi45LTM1LjYgNTguMy01MC42IDguNi0zLjcgMTgtNyAyNy43LTEwLjEgNS43IDE5LjYgMTMuMiA0MCAyMi41IDYwLjktOS4yIDIwLjgtMTYuNiA0MS4xLTIyLjIgNjAuNi05LjktMy4xLTE5LjMtNi41LTI4LTEwLjJ6TTMxMCA0OTBjLTEzLjYtNy44LTE5LjUtMzcuNS0xNC45LTc1LjcgMS4xLTkuNCAyLjktMTkuMyA1LjEtMjkuNCAxOS42IDQuOCA0MSA4LjUgNjMuNSAxMC45IDEzLjUgMTguNSAyNy41IDM1LjMgNDEuNiA1MC0zMi42IDMwLjMtNjMuMiA0Ni45LTg0IDQ2LjktNC41LS4xLTguMy0xLTExLjMtMi43em0yMzcuMi03Ni4yYzQuNyAzOC4yLTEuMSA2Ny45LTE0LjYgNzUuOC0zIDEuOC02LjkgMi42LTExLjUgMi42LTIwLjcgMC01MS40LTE2LjUtODQtNDYuNiAxNC0xNC43IDI4LTMxLjQgNDEuMy00OS45IDIyLjYtMi40IDQ0LTYuMSA2My42LTExIDIuMyAxMC4xIDQuMSAxOS44IDUuMiAyOS4xem0zOC41LTY2LjdjLTguNiAzLjctMTggNy0yNy43IDEwLjEtNS43LTE5LjYtMTMuMi00MC0yMi41LTYwLjkgOS4yLTIwLjggMTYuNi00MS4xIDIyLjItNjAuNiA5LjkgMy4xIDE5LjMgNi41IDI4LjEgMTAuMiAzNS40IDE1LjEgNTguMyAzNC45IDU4LjMgNTAuNi0uMSAxNS43LTIzIDM1LjYtNTguNCA1MC42ek0zMjAuOCA3OC40eiIvPgogICAgPGNpcmNsZSBjeD0iNDIwLjkiIGN5PSIyOTYuNSIgcj0iNDUuNyIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-redo: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgd2lkdGg9IjE2Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgICA8cGF0aCBkPSJNMCAwaDI0djI0SDB6IiBmaWxsPSJub25lIi8+PHBhdGggZD0iTTE4LjQgMTAuNkMxNi41NSA4Ljk5IDE0LjE1IDggMTEuNSA4Yy00LjY1IDAtOC41OCAzLjAzLTkuOTYgNy4yMkwzLjkgMTZjMS4wNS0zLjE5IDQuMDUtNS41IDcuNi01LjUgMS45NSAwIDMuNzMuNzIgNS4xMiAxLjg4TDEzIDE2aDlWN2wtMy42IDMuNnoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-refresh: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDE4IDE4Ij4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGggZD0iTTkgMTMuNWMtMi40OSAwLTQuNS0yLjAxLTQuNS00LjVTNi41MSA0LjUgOSA0LjVjMS4yNCAwIDIuMzYuNTIgMy4xNyAxLjMzTDEwIDhoNVYzbC0xLjc2IDEuNzZDMTIuMTUgMy42OCAxMC42NiAzIDkgMyA1LjY5IDMgMy4wMSA1LjY5IDMuMDEgOVM1LjY5IDE1IDkgMTVjMi45NyAwIDUuNDMtMi4xNiA1LjktNWgtMS41MmMtLjQ2IDItMi4yNCAzLjUtNC4zOCAzLjV6Ii8+CiAgICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-regex: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIwIDIwIj4KICA8ZyBjbGFzcz0ianAtaWNvbjIiIGZpbGw9IiM0MTQxNDEiPgogICAgPHJlY3QgeD0iMiIgeT0iMiIgd2lkdGg9IjE2IiBoZWlnaHQ9IjE2Ii8+CiAgPC9nPgoKICA8ZyBjbGFzcz0ianAtaWNvbi1hY2NlbnQyIiBmaWxsPSIjRkZGIj4KICAgIDxjaXJjbGUgY2xhc3M9InN0MiIgY3g9IjUuNSIgY3k9IjE0LjUiIHI9IjEuNSIvPgogICAgPHJlY3QgeD0iMTIiIHk9IjQiIGNsYXNzPSJzdDIiIHdpZHRoPSIxIiBoZWlnaHQ9IjgiLz4KICAgIDxyZWN0IHg9IjguNSIgeT0iNy41IiB0cmFuc2Zvcm09Im1hdHJpeCgwLjg2NiAtMC41IDAuNSAwLjg2NiAtMi4zMjU1IDcuMzIxOSkiIGNsYXNzPSJzdDIiIHdpZHRoPSI4IiBoZWlnaHQ9IjEiLz4KICAgIDxyZWN0IHg9IjEyIiB5PSI0IiB0cmFuc2Zvcm09Im1hdHJpeCgwLjUgLTAuODY2IDAuODY2IDAuNSAtMC42Nzc5IDE0LjgyNTIpIiBjbGFzcz0ic3QyIiB3aWR0aD0iMSIgaGVpZ2h0PSI4Ii8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-run: url(data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjI0IiB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGggZD0iTTggNXYxNGwxMS03eiIvPgogICAgPC9nPgo8L3N2Zz4K);
  --jp-icon-running: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDUxMiA1MTIiPgogIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICA8cGF0aCBkPSJNMjU2IDhDMTE5IDggOCAxMTkgOCAyNTZzMTExIDI0OCAyNDggMjQ4IDI0OC0xMTEgMjQ4LTI0OFMzOTMgOCAyNTYgOHptOTYgMzI4YzAgOC44LTcuMiAxNi0xNiAxNkgxNzZjLTguOCAwLTE2LTcuMi0xNi0xNlYxNzZjMC04LjggNy4yLTE2IDE2LTE2aDE2MGM4LjggMCAxNiA3LjIgMTYgMTZ2MTYweiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-save: url(data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjI0IiB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGggZD0iTTE3IDNINWMtMS4xMSAwLTIgLjktMiAydjE0YzAgMS4xLjg5IDIgMiAyaDE0YzEuMSAwIDItLjkgMi0yVjdsLTQtNHptLTUgMTZjLTEuNjYgMC0zLTEuMzQtMy0zczEuMzQtMyAzLTMgMyAxLjM0IDMgMy0xLjM0IDMtMyAzem0zLTEwSDVWNWgxMHY0eiIvPgogICAgPC9nPgo8L3N2Zz4K);
  --jp-icon-search: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMTggMTgiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTEyLjEsMTAuOWgtMC43bC0wLjItMC4yYzAuOC0wLjksMS4zLTIuMiwxLjMtMy41YzAtMy0yLjQtNS40LTUuNC01LjRTMS44LDQuMiwxLjgsNy4xczIuNCw1LjQsNS40LDUuNCBjMS4zLDAsMi41LTAuNSwzLjUtMS4zbDAuMiwwLjJ2MC43bDQuMSw0LjFsMS4yLTEuMkwxMi4xLDEwLjl6IE03LjEsMTAuOWMtMi4xLDAtMy43LTEuNy0zLjctMy43czEuNy0zLjcsMy43LTMuN3MzLjcsMS43LDMuNywzLjcgUzkuMiwxMC45LDcuMSwxMC45eiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-settings: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8cGF0aCBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIiBkPSJNMTkuNDMgMTIuOThjLjA0LS4zMi4wNy0uNjQuMDctLjk4cy0uMDMtLjY2LS4wNy0uOThsMi4xMS0xLjY1Yy4xOS0uMTUuMjQtLjQyLjEyLS42NGwtMi0zLjQ2Yy0uMTItLjIyLS4zOS0uMy0uNjEtLjIybC0yLjQ5IDFjLS41Mi0uNC0xLjA4LS43My0xLjY5LS45OGwtLjM4LTIuNjVBLjQ4OC40ODggMCAwMDE0IDJoLTRjLS4yNSAwLS40Ni4xOC0uNDkuNDJsLS4zOCAyLjY1Yy0uNjEuMjUtMS4xNy41OS0xLjY5Ljk4bC0yLjQ5LTFjLS4yMy0uMDktLjQ5IDAtLjYxLjIybC0yIDMuNDZjLS4xMy4yMi0uMDcuNDkuMTIuNjRsMi4xMSAxLjY1Yy0uMDQuMzItLjA3LjY1LS4wNy45OHMuMDMuNjYuMDcuOThsLTIuMTEgMS42NWMtLjE5LjE1LS4yNC40Mi0uMTIuNjRsMiAzLjQ2Yy4xMi4yMi4zOS4zLjYxLjIybDIuNDktMWMuNTIuNCAxLjA4LjczIDEuNjkuOThsLjM4IDIuNjVjLjAzLjI0LjI0LjQyLjQ5LjQyaDRjLjI1IDAgLjQ2LS4xOC40OS0uNDJsLjM4LTIuNjVjLjYxLS4yNSAxLjE3LS41OSAxLjY5LS45OGwyLjQ5IDFjLjIzLjA5LjQ5IDAgLjYxLS4yMmwyLTMuNDZjLjEyLS4yMi4wNy0uNDktLjEyLS42NGwtMi4xMS0xLjY1ek0xMiAxNS41Yy0xLjkzIDAtMy41LTEuNTctMy41LTMuNXMxLjU3LTMuNSAzLjUtMy41IDMuNSAxLjU3IDMuNSAzLjUtMS41NyAzLjUtMy41IDMuNXoiLz4KPC9zdmc+Cg==);
  --jp-icon-spreadsheet: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8cGF0aCBjbGFzcz0ianAtaWNvbi1jb250cmFzdDEganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNENBRjUwIiBkPSJNMi4yIDIuMnYxNy42aDE3LjZWMi4ySDIuMnptMTUuNCA3LjdoLTUuNVY0LjRoNS41djUuNXpNOS45IDQuNHY1LjVINC40VjQuNGg1LjV6bS01LjUgNy43aDUuNXY1LjVINC40di01LjV6bTcuNyA1LjV2LTUuNWg1LjV2NS41aC01LjV6Ii8+Cjwvc3ZnPgo=);
  --jp-icon-stop: url(data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjI0IiB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGggZD0iTTAgMGgyNHYyNEgweiIgZmlsbD0ibm9uZSIvPgogICAgICAgIDxwYXRoIGQ9Ik02IDZoMTJ2MTJINnoiLz4KICAgIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-tab: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTIxIDNIM2MtMS4xIDAtMiAuOS0yIDJ2MTRjMCAxLjEuOSAyIDIgMmgxOGMxLjEgMCAyLS45IDItMlY1YzAtMS4xLS45LTItMi0yem0wIDE2SDNWNWgxMHY0aDh2MTB6Ii8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-table-rows: url(data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjI0IiB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGggZD0iTTAgMGgyNHYyNEgweiIgZmlsbD0ibm9uZSIvPgogICAgICAgIDxwYXRoIGQ9Ik0yMSw4SDNWNGgxOFY4eiBNMjEsMTBIM3Y0aDE4VjEweiBNMjEsMTZIM3Y0aDE4VjE2eiIvPgogICAgPC9nPgo8L3N2Zz4=);
  --jp-icon-tag: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjgiIGhlaWdodD0iMjgiIHZpZXdCb3g9IjAgMCA0MyAyOCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KCTxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CgkJPHBhdGggZD0iTTI4LjgzMzIgMTIuMzM0TDMyLjk5OTggMTYuNTAwN0wzNy4xNjY1IDEyLjMzNEgyOC44MzMyWiIvPgoJCTxwYXRoIGQ9Ik0xNi4yMDk1IDIxLjYxMDRDMTUuNjg3MyAyMi4xMjk5IDE0Ljg0NDMgMjIuMTI5OSAxNC4zMjQ4IDIxLjYxMDRMNi45ODI5IDE0LjcyNDVDNi41NzI0IDE0LjMzOTQgNi4wODMxMyAxMy42MDk4IDYuMDQ3ODYgMTMuMDQ4MkM1Ljk1MzQ3IDExLjUyODggNi4wMjAwMiA4LjYxOTQ0IDYuMDY2MjEgNy4wNzY5NUM2LjA4MjgxIDYuNTE0NzcgNi41NTU0OCA2LjA0MzQ3IDcuMTE4MDQgNi4wMzA1NUM5LjA4ODYzIDUuOTg0NzMgMTMuMjYzOCA1LjkzNTc5IDEzLjY1MTggNi4zMjQyNUwyMS43MzY5IDEzLjYzOUMyMi4yNTYgMTQuMTU4NSAyMS43ODUxIDE1LjQ3MjQgMjEuMjYyIDE1Ljk5NDZMMTYuMjA5NSAyMS42MTA0Wk05Ljc3NTg1IDguMjY1QzkuMzM1NTEgNy44MjU2NiA4LjYyMzUxIDcuODI1NjYgOC4xODI4IDguMjY1QzcuNzQzNDYgOC43MDU3MSA3Ljc0MzQ2IDkuNDE3MzMgOC4xODI4IDkuODU2NjdDOC42MjM4MiAxMC4yOTY0IDkuMzM1ODIgMTAuMjk2NCA5Ljc3NTg1IDkuODU2NjdDMTAuMjE1NiA5LjQxNzMzIDEwLjIxNTYgOC43MDUzMyA5Ljc3NTg1IDguMjY1WiIvPgoJPC9nPgo8L3N2Zz4K);
  --jp-icon-terminal: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0IiA+CiAgICA8cmVjdCBjbGFzcz0ianAtaWNvbjIganAtaWNvbi1zZWxlY3RhYmxlIiB3aWR0aD0iMjAiIGhlaWdodD0iMjAiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDIgMikiIGZpbGw9IiMzMzMzMzMiLz4KICAgIDxwYXRoIGNsYXNzPSJqcC1pY29uLWFjY2VudDIganAtaWNvbi1zZWxlY3RhYmxlLWludmVyc2UiIGQ9Ik01LjA1NjY0IDguNzYxNzJDNS4wNTY2NCA4LjU5NzY2IDUuMDMxMjUgOC40NTMxMiA0Ljk4MDQ3IDguMzI4MTJDNC45MzM1OSA4LjE5OTIyIDQuODU1NDcgOC4wODIwMyA0Ljc0NjA5IDcuOTc2NTZDNC42NDA2MiA3Ljg3MTA5IDQuNSA3Ljc3NTM5IDQuMzI0MjIgNy42ODk0NUM0LjE1MjM0IDcuNTk5NjEgMy45NDMzNiA3LjUxMTcyIDMuNjk3MjcgNy40MjU3OEMzLjMwMjczIDcuMjg1MTYgMi45NDMzNiA3LjEzNjcyIDIuNjE5MTQgNi45ODA0N0MyLjI5NDkyIDYuODI0MjIgMi4wMTc1OCA2LjY0MjU4IDEuNzg3MTEgNi40MzU1NUMxLjU2MDU1IDYuMjI4NTIgMS4zODQ3NyA1Ljk4ODI4IDEuMjU5NzcgNS43MTQ4NEMxLjEzNDc3IDUuNDM3NSAxLjA3MjI3IDUuMTA5MzggMS4wNzIyNyA0LjczMDQ3QzEuMDcyMjcgNC4zOTg0NCAxLjEyODkxIDQuMDk1NyAxLjI0MjE5IDMuODIyMjdDMS4zNTU0NyAzLjU0NDkyIDEuNTE1NjIgMy4zMDQ2OSAxLjcyMjY2IDMuMTAxNTZDMS45Mjk2OSAyLjg5ODQ0IDIuMTc5NjkgMi43MzQzNyAyLjQ3MjY2IDIuNjA5MzhDMi43NjU2MiAyLjQ4NDM4IDMuMDkxOCAyLjQwNDMgMy40NTExNyAyLjM2OTE0VjEuMTA5MzhINC4zODg2N1YyLjM4MDg2QzQuNzQwMjMgMi40Mjc3MyA1LjA1NjY0IDIuNTIzNDQgNS4zMzc4OSAyLjY2Nzk3QzUuNjE5MTQgMi44MTI1IDUuODU3NDIgMy4wMDE5NSA2LjA1MjczIDMuMjM2MzNDNi4yNTE5NSAzLjQ2NjggNi40MDQzIDMuNzQwMjMgNi41MDk3NyA0LjA1NjY0QzYuNjE5MTQgNC4zNjkxNCA2LjY3MzgzIDQuNzIwNyA2LjY3MzgzIDUuMTExMzNINS4wNDQ5MkM1LjA0NDkyIDQuNjM4NjcgNC45Mzc1IDQuMjgxMjUgNC43MjI2NiA0LjAzOTA2QzQuNTA3ODEgMy43OTI5NyA0LjIxNjggMy42Njk5MiAzLjg0OTYxIDMuNjY5OTJDMy42NTAzOSAzLjY2OTkyIDMuNDc2NTYgMy42OTcyNyAzLjMyODEyIDMuNzUxOTVDMy4xODM1OSAzLjgwMjczIDMuMDY0NDUgMy44NzY5NSAyLjk3MDcgMy45NzQ2MUMyLjg3Njk1IDQuMDY4MzYgMi44MDY2NCA0LjE3OTY5IDIuNzU5NzcgNC4zMDg1OUMyLjcxNjggNC40Mzc1IDIuNjk1MzEgNC41NzgxMiAyLjY5NTMxIDQuNzMwNDdDMi42OTUzMSA0Ljg4MjgxIDIuNzE2OCA1LjAxOTUzIDIuNzU5NzcgNS4xNDA2MkMyLjgwNjY0IDUuMjU3ODEgMi44ODI4MSA1LjM2NzE5IDIuOTg4MjggNS40Njg3NUMzLjA5NzY2IDUuNTcwMzEgMy4yNDAyMyA1LjY2Nzk3IDMuNDE2MDIgNS43NjE3MkMzLjU5MTggNS44NTE1NiAzLjgxMDU1IDUuOTQzMzYgNC4wNzIyNyA2LjAzNzExQzQuNDY2OCA2LjE4NTU1IDQuODI0MjIgNi4zMzk4NCA1LjE0NDUzIDYuNUM1LjQ2NDg0IDYuNjU2MjUgNS43MzgyOCA2LjgzOTg0IDUuOTY0ODQgNy4wNTA3OEM2LjE5NTMxIDcuMjU3ODEgNi4zNzEwOSA3LjUgNi40OTIxOSA3Ljc3NzM0QzYuNjE3MTkgOC4wNTA3OCA2LjY3OTY5IDguMzc1IDYuNjc5NjkgOC43NUM2LjY3OTY5IDkuMDkzNzUgNi42MjMwNSA5LjQwNDMgNi41MDk3NyA5LjY4MTY0QzYuMzk2NDggOS45NTUwOCA2LjIzNDM4IDEwLjE5MTQgNi4wMjM0NCAxMC4zOTA2QzUuODEyNSAxMC41ODk4IDUuNTU4NTkgMTAuNzUgNS4yNjE3MiAxMC44NzExQzQuOTY0ODQgMTAuOTg4MyA0LjYzMjgxIDExLjA2NDUgNC4yNjU2MiAxMS4wOTk2VjEyLjI0OEgzLjMzMzk4VjExLjA5OTZDMy4wMDE5NSAxMS4wNjg0IDIuNjc5NjkgMTAuOTk2MSAyLjM2NzE5IDEwLjg4MjhDMi4wNTQ2OSAxMC43NjU2IDEuNzc3MzQgMTAuNTk3NyAxLjUzNTE2IDEwLjM3ODlDMS4yOTY4OCAxMC4xNjAyIDEuMTA1NDcgOS44ODQ3NyAwLjk2MDkzOCA5LjU1MjczQzAuODE2NDA2IDkuMjE2OCAwLjc0NDE0MSA4LjgxNDQ1IDAuNzQ0MTQxIDguMzQ1N0gyLjM3ODkxQzIuMzc4OTEgOC42MjY5NSAyLjQxOTkyIDguODYzMjggMi41MDE5NSA5LjA1NDY5QzIuNTgzOTggOS4yNDIxOSAyLjY4OTQ1IDkuMzkyNTggMi44MTgzNiA5LjUwNTg2QzIuOTUxMTcgOS42MTUyMyAzLjEwMTU2IDkuNjkzMzYgMy4yNjk1MyA5Ljc0MDIzQzMuNDM3NSA5Ljc4NzExIDMuNjA5MzggOS44MTA1NSAzLjc4NTE2IDkuODEwNTVDNC4yMDMxMiA5LjgxMDU1IDQuNTE5NTMgOS43MTI4OSA0LjczNDM4IDkuNTE3NThDNC45NDkyMiA5LjMyMjI3IDUuMDU2NjQgOS4wNzAzMSA1LjA1NjY0IDguNzYxNzJaTTEzLjQxOCAxMi4yNzE1SDguMDc0MjJWMTFIMTMuNDE4VjEyLjI3MTVaIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgzLjk1MjY0IDYpIiBmaWxsPSJ3aGl0ZSIvPgo8L3N2Zz4K);
  --jp-icon-text-editor: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8cGF0aCBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIiBkPSJNMTUgMTVIM3YyaDEydi0yem0wLThIM3YyaDEyVjd6TTMgMTNoMTh2LTJIM3Yyem0wIDhoMTh2LTJIM3Yyek0zIDN2MmgxOFYzSDN6Ii8+Cjwvc3ZnPgo=);
  --jp-icon-toc: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIj4KICAgIDxwYXRoIGQ9Ik03LDVIMjFWN0g3VjVNNywxM1YxMUgyMVYxM0g3TTQsNC41QTEuNSwxLjUgMCAwLDEgNS41LDZBMS41LDEuNSAwIDAsMSA0LDcuNUExLjUsMS41IDAgMCwxIDIuNSw2QTEuNSwxLjUgMCAwLDEgNCw0LjVNNCwxMC41QTEuNSwxLjUgMCAwLDEgNS41LDEyQTEuNSwxLjUgMCAwLDEgNCwxMy41QTEuNSwxLjUgMCAwLDEgMi41LDEyQTEuNSwxLjUgMCAwLDEgNCwxMC41TTcsMTlWMTdIMjFWMTlIN000LDE2LjVBMS41LDEuNSAwIDAsMSA1LjUsMThBMS41LDEuNSAwIDAsMSA0LDE5LjVBMS41LDEuNSAwIDAsMSAyLjUsMThBMS41LDEuNSAwIDAsMSA0LDE2LjVaIiAvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-tree-view: url(data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjI0IiB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGggZD0iTTAgMGgyNHYyNEgweiIgZmlsbD0ibm9uZSIvPgogICAgICAgIDxwYXRoIGQ9Ik0yMiAxMVYzaC03djNIOVYzSDJ2OGg3VjhoMnYxMGg0djNoN3YtOGgtN3YzaC0yVjhoMnYzeiIvPgogICAgPC9nPgo8L3N2Zz4=);
  --jp-icon-trusted: url(data:image/svg+xml;base64,PHN2ZyBmaWxsPSJub25lIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI1Ij4KICAgIDxwYXRoIGNsYXNzPSJqcC1pY29uMiIgc3Ryb2tlPSIjMzMzMzMzIiBzdHJva2Utd2lkdGg9IjIiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDIgMykiIGQ9Ik0xLjg2MDk0IDExLjQ0MDlDMC44MjY0NDggOC43NzAyNyAwLjg2Mzc3OSA2LjA1NzY0IDEuMjQ5MDcgNC4xOTkzMkMyLjQ4MjA2IDMuOTMzNDcgNC4wODA2OCAzLjQwMzQ3IDUuNjAxMDIgMi44NDQ5QzcuMjM1NDkgMi4yNDQ0IDguODU2NjYgMS41ODE1IDkuOTg3NiAxLjA5NTM5QzExLjA1OTcgMS41ODM0MSAxMi42MDk0IDIuMjQ0NCAxNC4yMTggMi44NDMzOUMxNS43NTAzIDMuNDEzOTQgMTcuMzk5NSAzLjk1MjU4IDE4Ljc1MzkgNC4yMTM4NUMxOS4xMzY0IDYuMDcxNzcgMTkuMTcwOSA4Ljc3NzIyIDE4LjEzOSAxMS40NDA5QzE3LjAzMDMgMTQuMzAzMiAxNC42NjY4IDE3LjE4NDQgOS45OTk5OSAxOC45MzU0QzUuMzMzMiAxNy4xODQ0IDIuOTY5NjggMTQuMzAzMiAxLjg2MDk0IDExLjQ0MDlaIi8+CiAgICA8cGF0aCBjbGFzcz0ianAtaWNvbjIiIGZpbGw9IiMzMzMzMzMiIHN0cm9rZT0iIzMzMzMzMyIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoOCA5Ljg2NzE5KSIgZD0iTTIuODYwMTUgNC44NjUzNUwwLjcyNjU0OSAyLjk5OTU5TDAgMy42MzA0NUwyLjg2MDE1IDYuMTMxNTdMOCAwLjYzMDg3Mkw3LjI3ODU3IDBMMi44NjAxNSA0Ljg2NTM1WiIvPgo8L3N2Zz4K);
  --jp-icon-undo: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTEyLjUgOGMtMi42NSAwLTUuMDUuOTktNi45IDIuNkwyIDd2OWg5bC0zLjYyLTMuNjJjMS4zOS0xLjE2IDMuMTYtMS44OCA1LjEyLTEuODggMy41NCAwIDYuNTUgMi4zMSA3LjYgNS41bDIuMzctLjc4QzIxLjA4IDExLjAzIDE3LjE1IDggMTIuNSA4eiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-vega: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8ZyBjbGFzcz0ianAtaWNvbjEganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjMjEyMTIxIj4KICAgIDxwYXRoIGQ9Ik0xMC42IDUuNGwyLjItMy4ySDIuMnY3LjNsNC02LjZ6Ii8+CiAgICA8cGF0aCBkPSJNMTUuOCAyLjJsLTQuNCA2LjZMNyA2LjNsLTQuOCA4djUuNWgxNy42VjIuMmgtNHptLTcgMTUuNEg1LjV2LTQuNGgzLjN2NC40em00LjQgMEg5LjhWOS44aDMuNHY3Ljh6bTQuNCAwaC0zLjRWNi41aDMuNHYxMS4xeiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-yaml: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8ZyBjbGFzcz0ianAtaWNvbi1jb250cmFzdDIganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjRDgxQjYwIj4KICAgIDxwYXRoIGQ9Ik03LjIgMTguNnYtNS40TDMgNS42aDMuM2wxLjQgMy4xYy4zLjkuNiAxLjYgMSAyLjUuMy0uOC42LTEuNiAxLTIuNWwxLjQtMy4xaDMuNGwtNC40IDcuNnY1LjVsLTIuOS0uMXoiLz4KICAgIDxjaXJjbGUgY2xhc3M9InN0MCIgY3g9IjE3LjYiIGN5PSIxNi41IiByPSIyLjEiLz4KICAgIDxjaXJjbGUgY2xhc3M9InN0MCIgY3g9IjE3LjYiIGN5PSIxMSIgcj0iMi4xIi8+CiAgPC9nPgo8L3N2Zz4K);
}

/* Icon CSS class declarations */

.jp-AddIcon {
  background-image: var(--jp-icon-add);
}
.jp-BugIcon {
  background-image: var(--jp-icon-bug);
}
.jp-BuildIcon {
  background-image: var(--jp-icon-build);
}
.jp-CaretDownEmptyIcon {
  background-image: var(--jp-icon-caret-down-empty);
}
.jp-CaretDownEmptyThinIcon {
  background-image: var(--jp-icon-caret-down-empty-thin);
}
.jp-CaretDownIcon {
  background-image: var(--jp-icon-caret-down);
}
.jp-CaretLeftIcon {
  background-image: var(--jp-icon-caret-left);
}
.jp-CaretRightIcon {
  background-image: var(--jp-icon-caret-right);
}
.jp-CaretUpEmptyThinIcon {
  background-image: var(--jp-icon-caret-up-empty-thin);
}
.jp-CaretUpIcon {
  background-image: var(--jp-icon-caret-up);
}
.jp-CaseSensitiveIcon {
  background-image: var(--jp-icon-case-sensitive);
}
.jp-CheckIcon {
  background-image: var(--jp-icon-check);
}
.jp-CircleEmptyIcon {
  background-image: var(--jp-icon-circle-empty);
}
.jp-CircleIcon {
  background-image: var(--jp-icon-circle);
}
.jp-ClearIcon {
  background-image: var(--jp-icon-clear);
}
.jp-CloseIcon {
  background-image: var(--jp-icon-close);
}
.jp-CodeIcon {
  background-image: var(--jp-icon-code);
}
.jp-ConsoleIcon {
  background-image: var(--jp-icon-console);
}
.jp-CopyIcon {
  background-image: var(--jp-icon-copy);
}
.jp-CopyrightIcon {
  background-image: var(--jp-icon-copyright);
}
.jp-CutIcon {
  background-image: var(--jp-icon-cut);
}
.jp-DownloadIcon {
  background-image: var(--jp-icon-download);
}
.jp-EditIcon {
  background-image: var(--jp-icon-edit);
}
.jp-EllipsesIcon {
  background-image: var(--jp-icon-ellipses);
}
.jp-ExtensionIcon {
  background-image: var(--jp-icon-extension);
}
.jp-FastForwardIcon {
  background-image: var(--jp-icon-fast-forward);
}
.jp-FileIcon {
  background-image: var(--jp-icon-file);
}
.jp-FileUploadIcon {
  background-image: var(--jp-icon-file-upload);
}
.jp-FilterListIcon {
  background-image: var(--jp-icon-filter-list);
}
.jp-FolderIcon {
  background-image: var(--jp-icon-folder);
}
.jp-Html5Icon {
  background-image: var(--jp-icon-html5);
}
.jp-ImageIcon {
  background-image: var(--jp-icon-image);
}
.jp-InspectorIcon {
  background-image: var(--jp-icon-inspector);
}
.jp-JsonIcon {
  background-image: var(--jp-icon-json);
}
.jp-JuliaIcon {
  background-image: var(--jp-icon-julia);
}
.jp-JupyterFaviconIcon {
  background-image: var(--jp-icon-jupyter-favicon);
}
.jp-JupyterIcon {
  background-image: var(--jp-icon-jupyter);
}
.jp-JupyterlabWordmarkIcon {
  background-image: var(--jp-icon-jupyterlab-wordmark);
}
.jp-KernelIcon {
  background-image: var(--jp-icon-kernel);
}
.jp-KeyboardIcon {
  background-image: var(--jp-icon-keyboard);
}
.jp-LauncherIcon {
  background-image: var(--jp-icon-launcher);
}
.jp-LineFormIcon {
  background-image: var(--jp-icon-line-form);
}
.jp-LinkIcon {
  background-image: var(--jp-icon-link);
}
.jp-ListIcon {
  background-image: var(--jp-icon-list);
}
.jp-ListingsInfoIcon {
  background-image: var(--jp-icon-listings-info);
}
.jp-MarkdownIcon {
  background-image: var(--jp-icon-markdown);
}
.jp-NewFolderIcon {
  background-image: var(--jp-icon-new-folder);
}
.jp-NotTrustedIcon {
  background-image: var(--jp-icon-not-trusted);
}
.jp-NotebookIcon {
  background-image: var(--jp-icon-notebook);
}
.jp-NumberingIcon {
  background-image: var(--jp-icon-numbering);
}
.jp-OfflineBoltIcon {
  background-image: var(--jp-icon-offline-bolt);
}
.jp-PaletteIcon {
  background-image: var(--jp-icon-palette);
}
.jp-PasteIcon {
  background-image: var(--jp-icon-paste);
}
.jp-PdfIcon {
  background-image: var(--jp-icon-pdf);
}
.jp-PythonIcon {
  background-image: var(--jp-icon-python);
}
.jp-RKernelIcon {
  background-image: var(--jp-icon-r-kernel);
}
.jp-ReactIcon {
  background-image: var(--jp-icon-react);
}
.jp-RedoIcon {
  background-image: var(--jp-icon-redo);
}
.jp-RefreshIcon {
  background-image: var(--jp-icon-refresh);
}
.jp-RegexIcon {
  background-image: var(--jp-icon-regex);
}
.jp-RunIcon {
  background-image: var(--jp-icon-run);
}
.jp-RunningIcon {
  background-image: var(--jp-icon-running);
}
.jp-SaveIcon {
  background-image: var(--jp-icon-save);
}
.jp-SearchIcon {
  background-image: var(--jp-icon-search);
}
.jp-SettingsIcon {
  background-image: var(--jp-icon-settings);
}
.jp-SpreadsheetIcon {
  background-image: var(--jp-icon-spreadsheet);
}
.jp-StopIcon {
  background-image: var(--jp-icon-stop);
}
.jp-TabIcon {
  background-image: var(--jp-icon-tab);
}
.jp-TableRowsIcon {
  background-image: var(--jp-icon-table-rows);
}
.jp-TagIcon {
  background-image: var(--jp-icon-tag);
}
.jp-TerminalIcon {
  background-image: var(--jp-icon-terminal);
}
.jp-TextEditorIcon {
  background-image: var(--jp-icon-text-editor);
}
.jp-TocIcon {
  background-image: var(--jp-icon-toc);
}
.jp-TreeViewIcon {
  background-image: var(--jp-icon-tree-view);
}
.jp-TrustedIcon {
  background-image: var(--jp-icon-trusted);
}
.jp-UndoIcon {
  background-image: var(--jp-icon-undo);
}
.jp-VegaIcon {
  background-image: var(--jp-icon-vega);
}
.jp-YamlIcon {
  background-image: var(--jp-icon-yaml);
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/**
 * (DEPRECATED) Support for consuming icons as CSS background images
 */

.jp-Icon,
.jp-MaterialIcon {
  background-position: center;
  background-repeat: no-repeat;
  background-size: 16px;
  min-width: 16px;
  min-height: 16px;
}

.jp-Icon-cover {
  background-position: center;
  background-repeat: no-repeat;
  background-size: cover;
}

/**
 * (DEPRECATED) Support for specific CSS icon sizes
 */

.jp-Icon-16 {
  background-size: 16px;
  min-width: 16px;
  min-height: 16px;
}

.jp-Icon-18 {
  background-size: 18px;
  min-width: 18px;
  min-height: 18px;
}

.jp-Icon-20 {
  background-size: 20px;
  min-width: 20px;
  min-height: 20px;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/**
 * Support for icons as inline SVG HTMLElements
 */

/* recolor the primary elements of an icon */
.jp-icon0[fill] {
  fill: var(--jp-inverse-layout-color0);
}
.jp-icon1[fill] {
  fill: var(--jp-inverse-layout-color1);
}
.jp-icon2[fill] {
  fill: var(--jp-inverse-layout-color2);
}
.jp-icon3[fill] {
  fill: var(--jp-inverse-layout-color3);
}
.jp-icon4[fill] {
  fill: var(--jp-inverse-layout-color4);
}

.jp-icon0[stroke] {
  stroke: var(--jp-inverse-layout-color0);
}
.jp-icon1[stroke] {
  stroke: var(--jp-inverse-layout-color1);
}
.jp-icon2[stroke] {
  stroke: var(--jp-inverse-layout-color2);
}
.jp-icon3[stroke] {
  stroke: var(--jp-inverse-layout-color3);
}
.jp-icon4[stroke] {
  stroke: var(--jp-inverse-layout-color4);
}
/* recolor the accent elements of an icon */
.jp-icon-accent0[fill] {
  fill: var(--jp-layout-color0);
}
.jp-icon-accent1[fill] {
  fill: var(--jp-layout-color1);
}
.jp-icon-accent2[fill] {
  fill: var(--jp-layout-color2);
}
.jp-icon-accent3[fill] {
  fill: var(--jp-layout-color3);
}
.jp-icon-accent4[fill] {
  fill: var(--jp-layout-color4);
}

.jp-icon-accent0[stroke] {
  stroke: var(--jp-layout-color0);
}
.jp-icon-accent1[stroke] {
  stroke: var(--jp-layout-color1);
}
.jp-icon-accent2[stroke] {
  stroke: var(--jp-layout-color2);
}
.jp-icon-accent3[stroke] {
  stroke: var(--jp-layout-color3);
}
.jp-icon-accent4[stroke] {
  stroke: var(--jp-layout-color4);
}
/* set the color of an icon to transparent */
.jp-icon-none[fill] {
  fill: none;
}

.jp-icon-none[stroke] {
  stroke: none;
}
/* brand icon colors. Same for light and dark */
.jp-icon-brand0[fill] {
  fill: var(--jp-brand-color0);
}
.jp-icon-brand1[fill] {
  fill: var(--jp-brand-color1);
}
.jp-icon-brand2[fill] {
  fill: var(--jp-brand-color2);
}
.jp-icon-brand3[fill] {
  fill: var(--jp-brand-color3);
}
.jp-icon-brand4[fill] {
  fill: var(--jp-brand-color4);
}

.jp-icon-brand0[stroke] {
  stroke: var(--jp-brand-color0);
}
.jp-icon-brand1[stroke] {
  stroke: var(--jp-brand-color1);
}
.jp-icon-brand2[stroke] {
  stroke: var(--jp-brand-color2);
}
.jp-icon-brand3[stroke] {
  stroke: var(--jp-brand-color3);
}
.jp-icon-brand4[stroke] {
  stroke: var(--jp-brand-color4);
}
/* warn icon colors. Same for light and dark */
.jp-icon-warn0[fill] {
  fill: var(--jp-warn-color0);
}
.jp-icon-warn1[fill] {
  fill: var(--jp-warn-color1);
}
.jp-icon-warn2[fill] {
  fill: var(--jp-warn-color2);
}
.jp-icon-warn3[fill] {
  fill: var(--jp-warn-color3);
}

.jp-icon-warn0[stroke] {
  stroke: var(--jp-warn-color0);
}
.jp-icon-warn1[stroke] {
  stroke: var(--jp-warn-color1);
}
.jp-icon-warn2[stroke] {
  stroke: var(--jp-warn-color2);
}
.jp-icon-warn3[stroke] {
  stroke: var(--jp-warn-color3);
}
/* icon colors that contrast well with each other and most backgrounds */
.jp-icon-contrast0[fill] {
  fill: var(--jp-icon-contrast-color0);
}
.jp-icon-contrast1[fill] {
  fill: var(--jp-icon-contrast-color1);
}
.jp-icon-contrast2[fill] {
  fill: var(--jp-icon-contrast-color2);
}
.jp-icon-contrast3[fill] {
  fill: var(--jp-icon-contrast-color3);
}

.jp-icon-contrast0[stroke] {
  stroke: var(--jp-icon-contrast-color0);
}
.jp-icon-contrast1[stroke] {
  stroke: var(--jp-icon-contrast-color1);
}
.jp-icon-contrast2[stroke] {
  stroke: var(--jp-icon-contrast-color2);
}
.jp-icon-contrast3[stroke] {
  stroke: var(--jp-icon-contrast-color3);
}

/* CSS for icons in selected items in the settings editor */
#setting-editor .jp-PluginList .jp-mod-selected .jp-icon-selectable[fill] {
  fill: #fff;
}
#setting-editor
  .jp-PluginList
  .jp-mod-selected
  .jp-icon-selectable-inverse[fill] {
  fill: var(--jp-brand-color1);
}

/* CSS for icons in selected filebrowser listing items */
.jp-DirListing-item.jp-mod-selected .jp-icon-selectable[fill] {
  fill: #fff;
}
.jp-DirListing-item.jp-mod-selected .jp-icon-selectable-inverse[fill] {
  fill: var(--jp-brand-color1);
}

/* CSS for icons in selected tabs in the sidebar tab manager */
#tab-manager .lm-TabBar-tab.jp-mod-active .jp-icon-selectable[fill] {
  fill: #fff;
}

#tab-manager .lm-TabBar-tab.jp-mod-active .jp-icon-selectable-inverse[fill] {
  fill: var(--jp-brand-color1);
}
#tab-manager
  .lm-TabBar-tab.jp-mod-active
  .jp-icon-hover
  :hover
  .jp-icon-selectable[fill] {
  fill: var(--jp-brand-color1);
}

#tab-manager
  .lm-TabBar-tab.jp-mod-active
  .jp-icon-hover
  :hover
  .jp-icon-selectable-inverse[fill] {
  fill: #fff;
}

/**
 * TODO: come up with non css-hack solution for showing the busy icon on top
 *  of the close icon
 * CSS for complex behavior of close icon of tabs in the sidebar tab manager
 */
#tab-manager
  .lm-TabBar-tab.jp-mod-dirty
  > .lm-TabBar-tabCloseIcon
  > :not(:hover)
  > .jp-icon3[fill] {
  fill: none;
}
#tab-manager
  .lm-TabBar-tab.jp-mod-dirty
  > .lm-TabBar-tabCloseIcon
  > :not(:hover)
  > .jp-icon-busy[fill] {
  fill: var(--jp-inverse-layout-color3);
}

#tab-manager
  .lm-TabBar-tab.jp-mod-dirty.jp-mod-active
  > .lm-TabBar-tabCloseIcon
  > :not(:hover)
  > .jp-icon-busy[fill] {
  fill: #fff;
}

/**
* TODO: come up with non css-hack solution for showing the busy icon on top
*  of the close icon
* CSS for complex behavior of close icon of tabs in the main area tabbar
*/
.lm-DockPanel-tabBar
  .lm-TabBar-tab.lm-mod-closable.jp-mod-dirty
  > .lm-TabBar-tabCloseIcon
  > :not(:hover)
  > .jp-icon3[fill] {
  fill: none;
}
.lm-DockPanel-tabBar
  .lm-TabBar-tab.lm-mod-closable.jp-mod-dirty
  > .lm-TabBar-tabCloseIcon
  > :not(:hover)
  > .jp-icon-busy[fill] {
  fill: var(--jp-inverse-layout-color3);
}

/* CSS for icons in status bar */
#jp-main-statusbar .jp-mod-selected .jp-icon-selectable[fill] {
  fill: #fff;
}

#jp-main-statusbar .jp-mod-selected .jp-icon-selectable-inverse[fill] {
  fill: var(--jp-brand-color1);
}
/* special handling for splash icon CSS. While the theme CSS reloads during
   splash, the splash icon can loose theming. To prevent that, we set a
   default for its color variable */
:root {
  --jp-warn-color0: var(--md-orange-700);
}

/* not sure what to do with this one, used in filebrowser listing */
.jp-DragIcon {
  margin-right: 4px;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/**
 * Support for alt colors for icons as inline SVG HTMLElements
 */

/* alt recolor the primary elements of an icon */
.jp-icon-alt .jp-icon0[fill] {
  fill: var(--jp-layout-color0);
}
.jp-icon-alt .jp-icon1[fill] {
  fill: var(--jp-layout-color1);
}
.jp-icon-alt .jp-icon2[fill] {
  fill: var(--jp-layout-color2);
}
.jp-icon-alt .jp-icon3[fill] {
  fill: var(--jp-layout-color3);
}
.jp-icon-alt .jp-icon4[fill] {
  fill: var(--jp-layout-color4);
}

.jp-icon-alt .jp-icon0[stroke] {
  stroke: var(--jp-layout-color0);
}
.jp-icon-alt .jp-icon1[stroke] {
  stroke: var(--jp-layout-color1);
}
.jp-icon-alt .jp-icon2[stroke] {
  stroke: var(--jp-layout-color2);
}
.jp-icon-alt .jp-icon3[stroke] {
  stroke: var(--jp-layout-color3);
}
.jp-icon-alt .jp-icon4[stroke] {
  stroke: var(--jp-layout-color4);
}

/* alt recolor the accent elements of an icon */
.jp-icon-alt .jp-icon-accent0[fill] {
  fill: var(--jp-inverse-layout-color0);
}
.jp-icon-alt .jp-icon-accent1[fill] {
  fill: var(--jp-inverse-layout-color1);
}
.jp-icon-alt .jp-icon-accent2[fill] {
  fill: var(--jp-inverse-layout-color2);
}
.jp-icon-alt .jp-icon-accent3[fill] {
  fill: var(--jp-inverse-layout-color3);
}
.jp-icon-alt .jp-icon-accent4[fill] {
  fill: var(--jp-inverse-layout-color4);
}

.jp-icon-alt .jp-icon-accent0[stroke] {
  stroke: var(--jp-inverse-layout-color0);
}
.jp-icon-alt .jp-icon-accent1[stroke] {
  stroke: var(--jp-inverse-layout-color1);
}
.jp-icon-alt .jp-icon-accent2[stroke] {
  stroke: var(--jp-inverse-layout-color2);
}
.jp-icon-alt .jp-icon-accent3[stroke] {
  stroke: var(--jp-inverse-layout-color3);
}
.jp-icon-alt .jp-icon-accent4[stroke] {
  stroke: var(--jp-inverse-layout-color4);
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-icon-hoverShow:not(:hover) svg {
  display: none !important;
}

/**
 * Support for hover colors for icons as inline SVG HTMLElements
 */

/**
 * regular colors
 */

/* recolor the primary elements of an icon */
.jp-icon-hover :hover .jp-icon0-hover[fill] {
  fill: var(--jp-inverse-layout-color0);
}
.jp-icon-hover :hover .jp-icon1-hover[fill] {
  fill: var(--jp-inverse-layout-color1);
}
.jp-icon-hover :hover .jp-icon2-hover[fill] {
  fill: var(--jp-inverse-layout-color2);
}
.jp-icon-hover :hover .jp-icon3-hover[fill] {
  fill: var(--jp-inverse-layout-color3);
}
.jp-icon-hover :hover .jp-icon4-hover[fill] {
  fill: var(--jp-inverse-layout-color4);
}

.jp-icon-hover :hover .jp-icon0-hover[stroke] {
  stroke: var(--jp-inverse-layout-color0);
}
.jp-icon-hover :hover .jp-icon1-hover[stroke] {
  stroke: var(--jp-inverse-layout-color1);
}
.jp-icon-hover :hover .jp-icon2-hover[stroke] {
  stroke: var(--jp-inverse-layout-color2);
}
.jp-icon-hover :hover .jp-icon3-hover[stroke] {
  stroke: var(--jp-inverse-layout-color3);
}
.jp-icon-hover :hover .jp-icon4-hover[stroke] {
  stroke: var(--jp-inverse-layout-color4);
}

/* recolor the accent elements of an icon */
.jp-icon-hover :hover .jp-icon-accent0-hover[fill] {
  fill: var(--jp-layout-color0);
}
.jp-icon-hover :hover .jp-icon-accent1-hover[fill] {
  fill: var(--jp-layout-color1);
}
.jp-icon-hover :hover .jp-icon-accent2-hover[fill] {
  fill: var(--jp-layout-color2);
}
.jp-icon-hover :hover .jp-icon-accent3-hover[fill] {
  fill: var(--jp-layout-color3);
}
.jp-icon-hover :hover .jp-icon-accent4-hover[fill] {
  fill: var(--jp-layout-color4);
}

.jp-icon-hover :hover .jp-icon-accent0-hover[stroke] {
  stroke: var(--jp-layout-color0);
}
.jp-icon-hover :hover .jp-icon-accent1-hover[stroke] {
  stroke: var(--jp-layout-color1);
}
.jp-icon-hover :hover .jp-icon-accent2-hover[stroke] {
  stroke: var(--jp-layout-color2);
}
.jp-icon-hover :hover .jp-icon-accent3-hover[stroke] {
  stroke: var(--jp-layout-color3);
}
.jp-icon-hover :hover .jp-icon-accent4-hover[stroke] {
  stroke: var(--jp-layout-color4);
}

/* set the color of an icon to transparent */
.jp-icon-hover :hover .jp-icon-none-hover[fill] {
  fill: none;
}

.jp-icon-hover :hover .jp-icon-none-hover[stroke] {
  stroke: none;
}

/**
 * inverse colors
 */

/* inverse recolor the primary elements of an icon */
.jp-icon-hover.jp-icon-alt :hover .jp-icon0-hover[fill] {
  fill: var(--jp-layout-color0);
}
.jp-icon-hover.jp-icon-alt :hover .jp-icon1-hover[fill] {
  fill: var(--jp-layout-color1);
}
.jp-icon-hover.jp-icon-alt :hover .jp-icon2-hover[fill] {
  fill: var(--jp-layout-color2);
}
.jp-icon-hover.jp-icon-alt :hover .jp-icon3-hover[fill] {
  fill: var(--jp-layout-color3);
}
.jp-icon-hover.jp-icon-alt :hover .jp-icon4-hover[fill] {
  fill: var(--jp-layout-color4);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon0-hover[stroke] {
  stroke: var(--jp-layout-color0);
}
.jp-icon-hover.jp-icon-alt :hover .jp-icon1-hover[stroke] {
  stroke: var(--jp-layout-color1);
}
.jp-icon-hover.jp-icon-alt :hover .jp-icon2-hover[stroke] {
  stroke: var(--jp-layout-color2);
}
.jp-icon-hover.jp-icon-alt :hover .jp-icon3-hover[stroke] {
  stroke: var(--jp-layout-color3);
}
.jp-icon-hover.jp-icon-alt :hover .jp-icon4-hover[stroke] {
  stroke: var(--jp-layout-color4);
}

/* inverse recolor the accent elements of an icon */
.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent0-hover[fill] {
  fill: var(--jp-inverse-layout-color0);
}
.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent1-hover[fill] {
  fill: var(--jp-inverse-layout-color1);
}
.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent2-hover[fill] {
  fill: var(--jp-inverse-layout-color2);
}
.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent3-hover[fill] {
  fill: var(--jp-inverse-layout-color3);
}
.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent4-hover[fill] {
  fill: var(--jp-inverse-layout-color4);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent0-hover[stroke] {
  stroke: var(--jp-inverse-layout-color0);
}
.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent1-hover[stroke] {
  stroke: var(--jp-inverse-layout-color1);
}
.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent2-hover[stroke] {
  stroke: var(--jp-inverse-layout-color2);
}
.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent3-hover[stroke] {
  stroke: var(--jp-inverse-layout-color3);
}
.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent4-hover[stroke] {
  stroke: var(--jp-inverse-layout-color4);
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-switch {
  display: flex;
  align-items: center;
  padding-left: 4px;
  padding-right: 4px;
  font-size: var(--jp-ui-font-size1);
  background-color: transparent;
  color: var(--jp-ui-font-color1);
  border: none;
  height: 20px;
}

.jp-switch:hover {
  background-color: var(--jp-layout-color2);
}

.jp-switch-label {
  margin-right: 5px;
}

.jp-switch-track {
  cursor: pointer;
  background-color: var(--jp-border-color1);
  -webkit-transition: 0.4s;
  transition: 0.4s;
  border-radius: 34px;
  height: 16px;
  width: 35px;
  position: relative;
}

.jp-switch-track::before {
  content: '';
  position: absolute;
  height: 10px;
  width: 10px;
  margin: 3px;
  left: 0px;
  background-color: var(--jp-ui-inverse-font-color1);
  -webkit-transition: 0.4s;
  transition: 0.4s;
  border-radius: 50%;
}

.jp-switch[aria-checked='true'] .jp-switch-track {
  background-color: var(--jp-warn-color0);
}

.jp-switch[aria-checked='true'] .jp-switch-track::before {
  /* track width (35) - margins (3 + 3) - thumb width (10) */
  left: 19px;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/* Sibling imports */

/* Override Blueprint's _reset.scss styles */
html {
  box-sizing: unset;
}

*,
*::before,
*::after {
  box-sizing: unset;
}

body {
  color: unset;
  font-family: var(--jp-ui-font-family);
}

p {
  margin-top: unset;
  margin-bottom: unset;
}

small {
  font-size: unset;
}

strong {
  font-weight: unset;
}

/* Override Blueprint's _typography.scss styles */
a {
  text-decoration: unset;
  color: unset;
}
a:hover {
  text-decoration: unset;
  color: unset;
}

/* Override Blueprint's _accessibility.scss styles */
:focus {
  outline: unset;
  outline-offset: unset;
  -moz-outline-radius: unset;
}

/* Styles for ui-components */
.jp-Button {
  border-radius: var(--jp-border-radius);
  padding: 0px 12px;
  font-size: var(--jp-ui-font-size1);
}

/* Use our own theme for hover styles */
button.jp-Button.bp3-button.bp3-minimal:hover {
  background-color: var(--jp-layout-color2);
}
.jp-Button.minimal {
  color: unset !important;
}

.jp-Button.jp-ToolbarButtonComponent {
  text-transform: none;
}

.jp-InputGroup input {
  box-sizing: border-box;
  border-radius: 0;
  background-color: transparent;
  color: var(--jp-ui-font-color0);
  box-shadow: inset 0 0 0 var(--jp-border-width) var(--jp-input-border-color);
}

.jp-InputGroup input:focus {
  box-shadow: inset 0 0 0 var(--jp-border-width)
      var(--jp-input-active-box-shadow-color),
    inset 0 0 0 3px var(--jp-input-active-box-shadow-color);
}

.jp-InputGroup input::placeholder,
input::placeholder {
  color: var(--jp-ui-font-color3);
}

.jp-BPIcon {
  display: inline-block;
  vertical-align: middle;
  margin: auto;
}

/* Stop blueprint futzing with our icon fills */
.bp3-icon.jp-BPIcon > svg:not([fill]) {
  fill: var(--jp-inverse-layout-color3);
}

.jp-InputGroupAction {
  padding: 6px;
}

.jp-HTMLSelect.jp-DefaultStyle select {
  background-color: initial;
  border: none;
  border-radius: 0;
  box-shadow: none;
  color: var(--jp-ui-font-color0);
  display: block;
  font-size: var(--jp-ui-font-size1);
  height: 24px;
  line-height: 14px;
  padding: 0 25px 0 10px;
  text-align: left;
  -moz-appearance: none;
  -webkit-appearance: none;
}

/* Use our own theme for hover and option styles */
.jp-HTMLSelect.jp-DefaultStyle select:hover,
.jp-HTMLSelect.jp-DefaultStyle select > option {
  background-color: var(--jp-layout-color2);
  color: var(--jp-ui-font-color0);
}
select {
  box-sizing: border-box;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-Collapse {
  display: flex;
  flex-direction: column;
  align-items: stretch;
  border-top: 1px solid var(--jp-border-color2);
  border-bottom: 1px solid var(--jp-border-color2);
}

.jp-Collapse-header {
  padding: 1px 12px;
  color: var(--jp-ui-font-color1);
  background-color: var(--jp-layout-color1);
  font-size: var(--jp-ui-font-size2);
}

.jp-Collapse-header:hover {
  background-color: var(--jp-layout-color2);
}

.jp-Collapse-contents {
  padding: 0px 12px 0px 12px;
  background-color: var(--jp-layout-color1);
  color: var(--jp-ui-font-color1);
  overflow: auto;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Variables
|----------------------------------------------------------------------------*/

:root {
  --jp-private-commandpalette-search-height: 28px;
}

/*-----------------------------------------------------------------------------
| Overall styles
|----------------------------------------------------------------------------*/

.lm-CommandPalette {
  padding-bottom: 0px;
  color: var(--jp-ui-font-color1);
  background: var(--jp-layout-color1);
  /* This is needed so that all font sizing of children done in ems is
   * relative to this base size */
  font-size: var(--jp-ui-font-size1);
}

/*-----------------------------------------------------------------------------
| Modal variant
|----------------------------------------------------------------------------*/

.jp-ModalCommandPalette {
  position: absolute;
  z-index: 10000;
  top: 38px;
  left: 30%;
  margin: 0;
  padding: 4px;
  width: 40%;
  box-shadow: var(--jp-elevation-z4);
  border-radius: 4px;
  background: var(--jp-layout-color0);
}

.jp-ModalCommandPalette .lm-CommandPalette {
  max-height: 40vh;
}

.jp-ModalCommandPalette .lm-CommandPalette .lm-close-icon::after {
  display: none;
}

.jp-ModalCommandPalette .lm-CommandPalette .lm-CommandPalette-header {
  display: none;
}

.jp-ModalCommandPalette .lm-CommandPalette .lm-CommandPalette-item {
  margin-left: 4px;
  margin-right: 4px;
}

.jp-ModalCommandPalette
  .lm-CommandPalette
  .lm-CommandPalette-item.lm-mod-disabled {
  display: none;
}

/*-----------------------------------------------------------------------------
| Search
|----------------------------------------------------------------------------*/

.lm-CommandPalette-search {
  padding: 4px;
  background-color: var(--jp-layout-color1);
  z-index: 2;
}

.lm-CommandPalette-wrapper {
  overflow: overlay;
  padding: 0px 9px;
  background-color: var(--jp-input-active-background);
  height: 30px;
  box-shadow: inset 0 0 0 var(--jp-border-width) var(--jp-input-border-color);
}

.lm-CommandPalette.lm-mod-focused .lm-CommandPalette-wrapper {
  box-shadow: inset 0 0 0 1px var(--jp-input-active-box-shadow-color),
    inset 0 0 0 3px var(--jp-input-active-box-shadow-color);
}

.jp-SearchIconGroup {
  color: white;
  background-color: var(--jp-brand-color1);
  position: absolute;
  top: 4px;
  right: 4px;
  padding: 5px 5px 1px 5px;
}

.jp-SearchIconGroup svg {
  height: 20px;
  width: 20px;
}

.jp-SearchIconGroup .jp-icon3[fill] {
  fill: var(--jp-layout-color0);
}

.lm-CommandPalette-input {
  background: transparent;
  width: calc(100% - 18px);
  float: left;
  border: none;
  outline: none;
  font-size: var(--jp-ui-font-size1);
  color: var(--jp-ui-font-color0);
  line-height: var(--jp-private-commandpalette-search-height);
}

.lm-CommandPalette-input::-webkit-input-placeholder,
.lm-CommandPalette-input::-moz-placeholder,
.lm-CommandPalette-input:-ms-input-placeholder {
  color: var(--jp-ui-font-color2);
  font-size: var(--jp-ui-font-size1);
}

/*-----------------------------------------------------------------------------
| Results
|----------------------------------------------------------------------------*/

.lm-CommandPalette-header:first-child {
  margin-top: 0px;
}

.lm-CommandPalette-header {
  border-bottom: solid var(--jp-border-width) var(--jp-border-color2);
  color: var(--jp-ui-font-color1);
  cursor: pointer;
  display: flex;
  font-size: var(--jp-ui-font-size0);
  font-weight: 600;
  letter-spacing: 1px;
  margin-top: 8px;
  padding: 8px 0 8px 12px;
  text-transform: uppercase;
}

.lm-CommandPalette-header.lm-mod-active {
  background: var(--jp-layout-color2);
}

.lm-CommandPalette-header > mark {
  background-color: transparent;
  font-weight: bold;
  color: var(--jp-ui-font-color1);
}

.lm-CommandPalette-item {
  padding: 4px 12px 4px 4px;
  color: var(--jp-ui-font-color1);
  font-size: var(--jp-ui-font-size1);
  font-weight: 400;
  display: flex;
}

.lm-CommandPalette-item.lm-mod-disabled {
  color: var(--jp-ui-font-color2);
}

.lm-CommandPalette-item.lm-mod-active {
  color: var(--jp-ui-inverse-font-color1);
  background: var(--jp-brand-color1);
}

.lm-CommandPalette-item.lm-mod-active .lm-CommandPalette-itemLabel > mark {
  color: var(--jp-ui-inverse-font-color0);
}

.lm-CommandPalette-item.lm-mod-active .jp-icon-selectable[fill] {
  fill: var(--jp-layout-color0);
}

.lm-CommandPalette-item.lm-mod-active .lm-CommandPalette-itemLabel > mark {
  color: var(--jp-ui-inverse-font-color0);
}

.lm-CommandPalette-item.lm-mod-active:hover:not(.lm-mod-disabled) {
  color: var(--jp-ui-inverse-font-color1);
  background: var(--jp-brand-color1);
}

.lm-CommandPalette-item:hover:not(.lm-mod-active):not(.lm-mod-disabled) {
  background: var(--jp-layout-color2);
}

.lm-CommandPalette-itemContent {
  overflow: hidden;
}

.lm-CommandPalette-itemLabel > mark {
  color: var(--jp-ui-font-color0);
  background-color: transparent;
  font-weight: bold;
}

.lm-CommandPalette-item.lm-mod-disabled mark {
  color: var(--jp-ui-font-color2);
}

.lm-CommandPalette-item .lm-CommandPalette-itemIcon {
  margin: 0 4px 0 0;
  position: relative;
  width: 16px;
  top: 2px;
  flex: 0 0 auto;
}

.lm-CommandPalette-item.lm-mod-disabled .lm-CommandPalette-itemIcon {
  opacity: 0.6;
}

.lm-CommandPalette-item .lm-CommandPalette-itemShortcut {
  flex: 0 0 auto;
}

.lm-CommandPalette-itemCaption {
  display: none;
}

.lm-CommandPalette-content {
  background-color: var(--jp-layout-color1);
}

.lm-CommandPalette-content:empty:after {
  content: 'No results';
  margin: auto;
  margin-top: 20px;
  width: 100px;
  display: block;
  font-size: var(--jp-ui-font-size2);
  font-family: var(--jp-ui-font-family);
  font-weight: lighter;
}

.lm-CommandPalette-emptyMessage {
  text-align: center;
  margin-top: 24px;
  line-height: 1.32;
  padding: 0px 8px;
  color: var(--jp-content-font-color3);
}

/*-----------------------------------------------------------------------------
| Copyright (c) 2014-2017, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-Dialog {
  position: absolute;
  z-index: 10000;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  top: 0px;
  left: 0px;
  margin: 0;
  padding: 0;
  width: 100%;
  height: 100%;
  background: var(--jp-dialog-background);
}

.jp-Dialog-content {
  display: flex;
  flex-direction: column;
  margin-left: auto;
  margin-right: auto;
  background: var(--jp-layout-color1);
  padding: 24px;
  padding-bottom: 12px;
  min-width: 300px;
  min-height: 150px;
  max-width: 1000px;
  max-height: 500px;
  box-sizing: border-box;
  box-shadow: var(--jp-elevation-z20);
  word-wrap: break-word;
  border-radius: var(--jp-border-radius);
  /* This is needed so that all font sizing of children done in ems is
   * relative to this base size */
  font-size: var(--jp-ui-font-size1);
  color: var(--jp-ui-font-color1);
  resize: both;
}

.jp-Dialog-button {
  overflow: visible;
}

button.jp-Dialog-button:focus {
  outline: 1px solid var(--jp-brand-color1);
  outline-offset: 4px;
  -moz-outline-radius: 0px;
}

button.jp-Dialog-button:focus::-moz-focus-inner {
  border: 0;
}

button.jp-Dialog-close-button {
  padding: 0;
  height: 100%;
  min-width: unset;
  min-height: unset;
}

.jp-Dialog-header {
  display: flex;
  justify-content: space-between;
  flex: 0 0 auto;
  padding-bottom: 12px;
  font-size: var(--jp-ui-font-size3);
  font-weight: 400;
  color: var(--jp-ui-font-color0);
}

.jp-Dialog-body {
  display: flex;
  flex-direction: column;
  flex: 1 1 auto;
  font-size: var(--jp-ui-font-size1);
  background: var(--jp-layout-color1);
  overflow: auto;
}

.jp-Dialog-footer {
  display: flex;
  flex-direction: row;
  justify-content: flex-end;
  flex: 0 0 auto;
  margin-left: -12px;
  margin-right: -12px;
  padding: 12px;
}

.jp-Dialog-title {
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
}

.jp-Dialog-body > .jp-select-wrapper {
  width: 100%;
}

.jp-Dialog-body > button {
  padding: 0px 16px;
}

.jp-Dialog-body > label {
  line-height: 1.4;
  color: var(--jp-ui-font-color0);
}

.jp-Dialog-button.jp-mod-styled:not(:last-child) {
  margin-right: 12px;
}

/*-----------------------------------------------------------------------------
| Copyright (c) 2014-2016, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-HoverBox {
  position: fixed;
}

.jp-HoverBox.jp-mod-outofview {
  display: none;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-IFrame {
  width: 100%;
  height: 100%;
}

.jp-IFrame > iframe {
  border: none;
}

/*
When drag events occur, `p-mod-override-cursor` is added to the body.
Because iframes steal all cursor events, the following two rules are necessary
to suppress pointer events while resize drags are occurring. There may be a
better solution to this problem.
*/
body.lm-mod-override-cursor .jp-IFrame {
  position: relative;
}

body.lm-mod-override-cursor .jp-IFrame:before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: transparent;
}

.jp-Input-Boolean-Dialog {
  flex-direction: row-reverse;
  align-items: end;
  width: 100%;
}

.jp-Input-Boolean-Dialog > label {
  flex: 1 1 auto;
}

/*-----------------------------------------------------------------------------
| Copyright (c) 2014-2016, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-MainAreaWidget > :focus {
  outline: none;
}

/**
 * google-material-color v1.2.6
 * https://github.com/danlevan/google-material-color
 */
:root {
  --md-red-50: #ffebee;
  --md-red-100: #ffcdd2;
  --md-red-200: #ef9a9a;
  --md-red-300: #e57373;
  --md-red-400: #ef5350;
  --md-red-500: #f44336;
  --md-red-600: #e53935;
  --md-red-700: #d32f2f;
  --md-red-800: #c62828;
  --md-red-900: #b71c1c;
  --md-red-A100: #ff8a80;
  --md-red-A200: #ff5252;
  --md-red-A400: #ff1744;
  --md-red-A700: #d50000;

  --md-pink-50: #fce4ec;
  --md-pink-100: #f8bbd0;
  --md-pink-200: #f48fb1;
  --md-pink-300: #f06292;
  --md-pink-400: #ec407a;
  --md-pink-500: #e91e63;
  --md-pink-600: #d81b60;
  --md-pink-700: #c2185b;
  --md-pink-800: #ad1457;
  --md-pink-900: #880e4f;
  --md-pink-A100: #ff80ab;
  --md-pink-A200: #ff4081;
  --md-pink-A400: #f50057;
  --md-pink-A700: #c51162;

  --md-purple-50: #f3e5f5;
  --md-purple-100: #e1bee7;
  --md-purple-200: #ce93d8;
  --md-purple-300: #ba68c8;
  --md-purple-400: #ab47bc;
  --md-purple-500: #9c27b0;
  --md-purple-600: #8e24aa;
  --md-purple-700: #7b1fa2;
  --md-purple-800: #6a1b9a;
  --md-purple-900: #4a148c;
  --md-purple-A100: #ea80fc;
  --md-purple-A200: #e040fb;
  --md-purple-A400: #d500f9;
  --md-purple-A700: #aa00ff;

  --md-deep-purple-50: #ede7f6;
  --md-deep-purple-100: #d1c4e9;
  --md-deep-purple-200: #b39ddb;
  --md-deep-purple-300: #9575cd;
  --md-deep-purple-400: #7e57c2;
  --md-deep-purple-500: #673ab7;
  --md-deep-purple-600: #5e35b1;
  --md-deep-purple-700: #512da8;
  --md-deep-purple-800: #4527a0;
  --md-deep-purple-900: #311b92;
  --md-deep-purple-A100: #b388ff;
  --md-deep-purple-A200: #7c4dff;
  --md-deep-purple-A400: #651fff;
  --md-deep-purple-A700: #6200ea;

  --md-indigo-50: #e8eaf6;
  --md-indigo-100: #c5cae9;
  --md-indigo-200: #9fa8da;
  --md-indigo-300: #7986cb;
  --md-indigo-400: #5c6bc0;
  --md-indigo-500: #3f51b5;
  --md-indigo-600: #3949ab;
  --md-indigo-700: #303f9f;
  --md-indigo-800: #283593;
  --md-indigo-900: #1a237e;
  --md-indigo-A100: #8c9eff;
  --md-indigo-A200: #536dfe;
  --md-indigo-A400: #3d5afe;
  --md-indigo-A700: #304ffe;

  --md-blue-50: #e3f2fd;
  --md-blue-100: #bbdefb;
  --md-blue-200: #90caf9;
  --md-blue-300: #64b5f6;
  --md-blue-400: #42a5f5;
  --md-blue-500: #2196f3;
  --md-blue-600: #1e88e5;
  --md-blue-700: #1976d2;
  --md-blue-800: #1565c0;
  --md-blue-900: #0d47a1;
  --md-blue-A100: #82b1ff;
  --md-blue-A200: #448aff;
  --md-blue-A400: #2979ff;
  --md-blue-A700: #2962ff;

  --md-light-blue-50: #e1f5fe;
  --md-light-blue-100: #b3e5fc;
  --md-light-blue-200: #81d4fa;
  --md-light-blue-300: #4fc3f7;
  --md-light-blue-400: #29b6f6;
  --md-light-blue-500: #03a9f4;
  --md-light-blue-600: #039be5;
  --md-light-blue-700: #0288d1;
  --md-light-blue-800: #0277bd;
  --md-light-blue-900: #01579b;
  --md-light-blue-A100: #80d8ff;
  --md-light-blue-A200: #40c4ff;
  --md-light-blue-A400: #00b0ff;
  --md-light-blue-A700: #0091ea;

  --md-cyan-50: #e0f7fa;
  --md-cyan-100: #b2ebf2;
  --md-cyan-200: #80deea;
  --md-cyan-300: #4dd0e1;
  --md-cyan-400: #26c6da;
  --md-cyan-500: #00bcd4;
  --md-cyan-600: #00acc1;
  --md-cyan-700: #0097a7;
  --md-cyan-800: #00838f;
  --md-cyan-900: #006064;
  --md-cyan-A100: #84ffff;
  --md-cyan-A200: #18ffff;
  --md-cyan-A400: #00e5ff;
  --md-cyan-A700: #00b8d4;

  --md-teal-50: #e0f2f1;
  --md-teal-100: #b2dfdb;
  --md-teal-200: #80cbc4;
  --md-teal-300: #4db6ac;
  --md-teal-400: #26a69a;
  --md-teal-500: #009688;
  --md-teal-600: #00897b;
  --md-teal-700: #00796b;
  --md-teal-800: #00695c;
  --md-teal-900: #004d40;
  --md-teal-A100: #a7ffeb;
  --md-teal-A200: #64ffda;
  --md-teal-A400: #1de9b6;
  --md-teal-A700: #00bfa5;

  --md-green-50: #e8f5e9;
  --md-green-100: #c8e6c9;
  --md-green-200: #a5d6a7;
  --md-green-300: #81c784;
  --md-green-400: #66bb6a;
  --md-green-500: #4caf50;
  --md-green-600: #43a047;
  --md-green-700: #388e3c;
  --md-green-800: #2e7d32;
  --md-green-900: #1b5e20;
  --md-green-A100: #b9f6ca;
  --md-green-A200: #69f0ae;
  --md-green-A400: #00e676;
  --md-green-A700: #00c853;

  --md-light-green-50: #f1f8e9;
  --md-light-green-100: #dcedc8;
  --md-light-green-200: #c5e1a5;
  --md-light-green-300: #aed581;
  --md-light-green-400: #9ccc65;
  --md-light-green-500: #8bc34a;
  --md-light-green-600: #7cb342;
  --md-light-green-700: #689f38;
  --md-light-green-800: #558b2f;
  --md-light-green-900: #33691e;
  --md-light-green-A100: #ccff90;
  --md-light-green-A200: #b2ff59;
  --md-light-green-A400: #76ff03;
  --md-light-green-A700: #64dd17;

  --md-lime-50: #f9fbe7;
  --md-lime-100: #f0f4c3;
  --md-lime-200: #e6ee9c;
  --md-lime-300: #dce775;
  --md-lime-400: #d4e157;
  --md-lime-500: #cddc39;
  --md-lime-600: #c0ca33;
  --md-lime-700: #afb42b;
  --md-lime-800: #9e9d24;
  --md-lime-900: #827717;
  --md-lime-A100: #f4ff81;
  --md-lime-A200: #eeff41;
  --md-lime-A400: #c6ff00;
  --md-lime-A700: #aeea00;

  --md-yellow-50: #fffde7;
  --md-yellow-100: #fff9c4;
  --md-yellow-200: #fff59d;
  --md-yellow-300: #fff176;
  --md-yellow-400: #ffee58;
  --md-yellow-500: #ffeb3b;
  --md-yellow-600: #fdd835;
  --md-yellow-700: #fbc02d;
  --md-yellow-800: #f9a825;
  --md-yellow-900: #f57f17;
  --md-yellow-A100: #ffff8d;
  --md-yellow-A200: #ffff00;
  --md-yellow-A400: #ffea00;
  --md-yellow-A700: #ffd600;

  --md-amber-50: #fff8e1;
  --md-amber-100: #ffecb3;
  --md-amber-200: #ffe082;
  --md-amber-300: #ffd54f;
  --md-amber-400: #ffca28;
  --md-amber-500: #ffc107;
  --md-amber-600: #ffb300;
  --md-amber-700: #ffa000;
  --md-amber-800: #ff8f00;
  --md-amber-900: #ff6f00;
  --md-amber-A100: #ffe57f;
  --md-amber-A200: #ffd740;
  --md-amber-A400: #ffc400;
  --md-amber-A700: #ffab00;

  --md-orange-50: #fff3e0;
  --md-orange-100: #ffe0b2;
  --md-orange-200: #ffcc80;
  --md-orange-300: #ffb74d;
  --md-orange-400: #ffa726;
  --md-orange-500: #ff9800;
  --md-orange-600: #fb8c00;
  --md-orange-700: #f57c00;
  --md-orange-800: #ef6c00;
  --md-orange-900: #e65100;
  --md-orange-A100: #ffd180;
  --md-orange-A200: #ffab40;
  --md-orange-A400: #ff9100;
  --md-orange-A700: #ff6d00;

  --md-deep-orange-50: #fbe9e7;
  --md-deep-orange-100: #ffccbc;
  --md-deep-orange-200: #ffab91;
  --md-deep-orange-300: #ff8a65;
  --md-deep-orange-400: #ff7043;
  --md-deep-orange-500: #ff5722;
  --md-deep-orange-600: #f4511e;
  --md-deep-orange-700: #e64a19;
  --md-deep-orange-800: #d84315;
  --md-deep-orange-900: #bf360c;
  --md-deep-orange-A100: #ff9e80;
  --md-deep-orange-A200: #ff6e40;
  --md-deep-orange-A400: #ff3d00;
  --md-deep-orange-A700: #dd2c00;

  --md-brown-50: #efebe9;
  --md-brown-100: #d7ccc8;
  --md-brown-200: #bcaaa4;
  --md-brown-300: #a1887f;
  --md-brown-400: #8d6e63;
  --md-brown-500: #795548;
  --md-brown-600: #6d4c41;
  --md-brown-700: #5d4037;
  --md-brown-800: #4e342e;
  --md-brown-900: #3e2723;

  --md-grey-50: #fafafa;
  --md-grey-100: #f5f5f5;
  --md-grey-200: #eeeeee;
  --md-grey-300: #e0e0e0;
  --md-grey-400: #bdbdbd;
  --md-grey-500: #9e9e9e;
  --md-grey-600: #757575;
  --md-grey-700: #616161;
  --md-grey-800: #424242;
  --md-grey-900: #212121;

  --md-blue-grey-50: #eceff1;
  --md-blue-grey-100: #cfd8dc;
  --md-blue-grey-200: #b0bec5;
  --md-blue-grey-300: #90a4ae;
  --md-blue-grey-400: #78909c;
  --md-blue-grey-500: #607d8b;
  --md-blue-grey-600: #546e7a;
  --md-blue-grey-700: #455a64;
  --md-blue-grey-800: #37474f;
  --md-blue-grey-900: #263238;
}

/*-----------------------------------------------------------------------------
| Copyright (c) 2017, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-Spinner {
  position: absolute;
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 10;
  left: 0;
  top: 0;
  width: 100%;
  height: 100%;
  background: var(--jp-layout-color0);
  outline: none;
}

.jp-SpinnerContent {
  font-size: 10px;
  margin: 50px auto;
  text-indent: -9999em;
  width: 3em;
  height: 3em;
  border-radius: 50%;
  background: var(--jp-brand-color3);
  background: linear-gradient(
    to right,
    #f37626 10%,
    rgba(255, 255, 255, 0) 42%
  );
  position: relative;
  animation: load3 1s infinite linear, fadeIn 1s;
}

.jp-SpinnerContent:before {
  width: 50%;
  height: 50%;
  background: #f37626;
  border-radius: 100% 0 0 0;
  position: absolute;
  top: 0;
  left: 0;
  content: '';
}

.jp-SpinnerContent:after {
  background: var(--jp-layout-color0);
  width: 75%;
  height: 75%;
  border-radius: 50%;
  content: '';
  margin: auto;
  position: absolute;
  top: 0;
  left: 0;
  bottom: 0;
  right: 0;
}

@keyframes fadeIn {
  0% {
    opacity: 0;
  }
  100% {
    opacity: 1;
  }
}

@keyframes load3 {
  0% {
    transform: rotate(0deg);
  }
  100% {
    transform: rotate(360deg);
  }
}

/*-----------------------------------------------------------------------------
| Copyright (c) 2014-2017, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

button.jp-mod-styled {
  font-size: var(--jp-ui-font-size1);
  color: var(--jp-ui-font-color0);
  border: none;
  box-sizing: border-box;
  text-align: center;
  line-height: 32px;
  height: 32px;
  padding: 0px 12px;
  letter-spacing: 0.8px;
  outline: none;
  appearance: none;
  -webkit-appearance: none;
  -moz-appearance: none;
}

input.jp-mod-styled {
  background: var(--jp-input-background);
  height: 28px;
  box-sizing: border-box;
  border: var(--jp-border-width) solid var(--jp-border-color1);
  padding-left: 7px;
  padding-right: 7px;
  font-size: var(--jp-ui-font-size2);
  color: var(--jp-ui-font-color0);
  outline: none;
  appearance: none;
  -webkit-appearance: none;
  -moz-appearance: none;
}

input[type='checkbox'].jp-mod-styled {
  appearance: checkbox;
  -webkit-appearance: checkbox;
  -moz-appearance: checkbox;
  height: auto;
}

input.jp-mod-styled:focus {
  border: var(--jp-border-width) solid var(--md-blue-500);
  box-shadow: inset 0 0 4px var(--md-blue-300);
}

.jp-FileDialog-Checkbox {
  margin-top: 35px;
  display: flex;
  flex-direction: row;
  align-items: end;
  width: 100%;
}

.jp-FileDialog-Checkbox > label {
  flex: 1 1 auto;
}

.jp-select-wrapper {
  display: flex;
  position: relative;
  flex-direction: column;
  padding: 1px;
  background-color: var(--jp-layout-color1);
  height: 28px;
  box-sizing: border-box;
  margin-bottom: 12px;
}

.jp-select-wrapper.jp-mod-focused select.jp-mod-styled {
  border: var(--jp-border-width) solid var(--jp-input-active-border-color);
  box-shadow: var(--jp-input-box-shadow);
  background-color: var(--jp-input-active-background);
}

select.jp-mod-styled:hover {
  background-color: var(--jp-layout-color1);
  cursor: pointer;
  color: var(--jp-ui-font-color0);
  background-color: var(--jp-input-hover-background);
  box-shadow: inset 0 0px 1px rgba(0, 0, 0, 0.5);
}

select.jp-mod-styled {
  flex: 1 1 auto;
  height: 32px;
  width: 100%;
  font-size: var(--jp-ui-font-size2);
  background: var(--jp-input-background);
  color: var(--jp-ui-font-color0);
  padding: 0 25px 0 8px;
  border: var(--jp-border-width) solid var(--jp-input-border-color);
  border-radius: 0px;
  outline: none;
  appearance: none;
  -webkit-appearance: none;
  -moz-appearance: none;
}

/*-----------------------------------------------------------------------------
| Copyright (c) 2014-2016, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

:root {
  --jp-private-toolbar-height: calc(
    28px + var(--jp-border-width)
  ); /* leave 28px for content */
}

.jp-Toolbar {
  color: var(--jp-ui-font-color1);
  flex: 0 0 auto;
  display: flex;
  flex-direction: row;
  border-bottom: var(--jp-border-width) solid var(--jp-toolbar-border-color);
  box-shadow: var(--jp-toolbar-box-shadow);
  background: var(--jp-toolbar-background);
  min-height: var(--jp-toolbar-micro-height);
  padding: 2px;
  z-index: 1;
  overflow-x: auto;
}

/* Toolbar items */

.jp-Toolbar > .jp-Toolbar-item.jp-Toolbar-spacer {
  flex-grow: 1;
  flex-shrink: 1;
}

.jp-Toolbar-item.jp-Toolbar-kernelStatus {
  display: inline-block;
  width: 32px;
  background-repeat: no-repeat;
  background-position: center;
  background-size: 16px;
}

.jp-Toolbar > .jp-Toolbar-item {
  flex: 0 0 auto;
  display: flex;
  padding-left: 1px;
  padding-right: 1px;
  font-size: var(--jp-ui-font-size1);
  line-height: var(--jp-private-toolbar-height);
  height: 100%;
}

/* Toolbar buttons */

/* This is the div we use to wrap the react component into a Widget */
div.jp-ToolbarButton {
  color: transparent;
  border: none;
  box-sizing: border-box;
  outline: none;
  appearance: none;
  -webkit-appearance: none;
  -moz-appearance: none;
  padding: 0px;
  margin: 0px;
}

button.jp-ToolbarButtonComponent {
  background: var(--jp-layout-color1);
  border: none;
  box-sizing: border-box;
  outline: none;
  appearance: none;
  -webkit-appearance: none;
  -moz-appearance: none;
  padding: 0px 6px;
  margin: 0px;
  height: 24px;
  border-radius: var(--jp-border-radius);
  display: flex;
  align-items: center;
  text-align: center;
  font-size: 14px;
  min-width: unset;
  min-height: unset;
}

button.jp-ToolbarButtonComponent:disabled {
  opacity: 0.4;
}

button.jp-ToolbarButtonComponent span {
  padding: 0px;
  flex: 0 0 auto;
}

button.jp-ToolbarButtonComponent .jp-ToolbarButtonComponent-label {
  font-size: var(--jp-ui-font-size1);
  line-height: 100%;
  padding-left: 2px;
  color: var(--jp-ui-font-color1);
}

#jp-main-dock-panel[data-mode='single-document']
  .jp-MainAreaWidget
  > .jp-Toolbar.jp-Toolbar-micro {
  padding: 0;
  min-height: 0;
}

#jp-main-dock-panel[data-mode='single-document']
  .jp-MainAreaWidget
  > .jp-Toolbar {
  border: none;
  box-shadow: none;
}

/*-----------------------------------------------------------------------------
| Copyright (c) 2014-2017, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/


/* <DEPRECATED> */ body.p-mod-override-cursor *, /* </DEPRECATED> */
body.lm-mod-override-cursor * {
  cursor: inherit !important;
}

/*-----------------------------------------------------------------------------
| Copyright (c) 2014-2016, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-JSONEditor {
  display: flex;
  flex-direction: column;
  width: 100%;
}

.jp-JSONEditor-host {
  flex: 1 1 auto;
  border: var(--jp-border-width) solid var(--jp-input-border-color);
  border-radius: 0px;
  background: var(--jp-layout-color0);
  min-height: 50px;
  padding: 1px;
}

.jp-JSONEditor.jp-mod-error .jp-JSONEditor-host {
  border-color: red;
  outline-color: red;
}

.jp-JSONEditor-header {
  display: flex;
  flex: 1 0 auto;
  padding: 0 0 0 12px;
}

.jp-JSONEditor-header label {
  flex: 0 0 auto;
}

.jp-JSONEditor-commitButton {
  height: 16px;
  width: 16px;
  background-size: 18px;
  background-repeat: no-repeat;
  background-position: center;
}

.jp-JSONEditor-host.jp-mod-focused {
  background-color: var(--jp-input-active-background);
  border: 1px solid var(--jp-input-active-border-color);
  box-shadow: var(--jp-input-box-shadow);
}

.jp-Editor.jp-mod-dropTarget {
  border: var(--jp-border-width) solid var(--jp-input-active-border-color);
  box-shadow: var(--jp-input-box-shadow);
}

/* BASICS */

.CodeMirror {
  /* Set height, width, borders, and global font properties here */
  font-family: monospace;
  height: 300px;
  color: black;
  direction: ltr;
}

/* PADDING */

.CodeMirror-lines {
  padding: 4px 0; /* Vertical padding around content */
}
.CodeMirror pre.CodeMirror-line,
.CodeMirror pre.CodeMirror-line-like {
  padding: 0 4px; /* Horizontal padding of content */
}

.CodeMirror-scrollbar-filler, .CodeMirror-gutter-filler {
  background-color: white; /* The little square between H and V scrollbars */
}

/* GUTTER */

.CodeMirror-gutters {
  border-right: 1px solid #ddd;
  background-color: #f7f7f7;
  white-space: nowrap;
}
.CodeMirror-linenumbers {}
.CodeMirror-linenumber {
  padding: 0 3px 0 5px;
  min-width: 20px;
  text-align: right;
  color: #999;
  white-space: nowrap;
}

.CodeMirror-guttermarker { color: black; }
.CodeMirror-guttermarker-subtle { color: #999; }

/* CURSOR */

.CodeMirror-cursor {
  border-left: 1px solid black;
  border-right: none;
  width: 0;
}
/* Shown when moving in bi-directional text */
.CodeMirror div.CodeMirror-secondarycursor {
  border-left: 1px solid silver;
}
.cm-fat-cursor .CodeMirror-cursor {
  width: auto;
  border: 0 !important;
  background: #7e7;
}
.cm-fat-cursor div.CodeMirror-cursors {
  z-index: 1;
}
.cm-fat-cursor-mark {
  background-color: rgba(20, 255, 20, 0.5);
  -webkit-animation: blink 1.06s steps(1) infinite;
  -moz-animation: blink 1.06s steps(1) infinite;
  animation: blink 1.06s steps(1) infinite;
}
.cm-animate-fat-cursor {
  width: auto;
  border: 0;
  -webkit-animation: blink 1.06s steps(1) infinite;
  -moz-animation: blink 1.06s steps(1) infinite;
  animation: blink 1.06s steps(1) infinite;
  background-color: #7e7;
}
@-moz-keyframes blink {
  0% {}
  50% { background-color: transparent; }
  100% {}
}
@-webkit-keyframes blink {
  0% {}
  50% { background-color: transparent; }
  100% {}
}
@keyframes blink {
  0% {}
  50% { background-color: transparent; }
  100% {}
}

/* Can style cursor different in overwrite (non-insert) mode */
.CodeMirror-overwrite .CodeMirror-cursor {}

.cm-tab { display: inline-block; text-decoration: inherit; }

.CodeMirror-rulers {
  position: absolute;
  left: 0; right: 0; top: -50px; bottom: 0;
  overflow: hidden;
}
.CodeMirror-ruler {
  border-left: 1px solid #ccc;
  top: 0; bottom: 0;
  position: absolute;
}

/* DEFAULT THEME */

.cm-s-default .cm-header {color: blue;}
.cm-s-default .cm-quote {color: #090;}
.cm-negative {color: #d44;}
.cm-positive {color: #292;}
.cm-header, .cm-strong {font-weight: bold;}
.cm-em {font-style: italic;}
.cm-link {text-decoration: underline;}
.cm-strikethrough {text-decoration: line-through;}

.cm-s-default .cm-keyword {color: #708;}
.cm-s-default .cm-atom {color: #219;}
.cm-s-default .cm-number {color: #164;}
.cm-s-default .cm-def {color: #00f;}
.cm-s-default .cm-variable,
.cm-s-default .cm-punctuation,
.cm-s-default .cm-property,
.cm-s-default .cm-operator {}
.cm-s-default .cm-variable-2 {color: #05a;}
.cm-s-default .cm-variable-3, .cm-s-default .cm-type {color: #085;}
.cm-s-default .cm-comment {color: #a50;}
.cm-s-default .cm-string {color: #a11;}
.cm-s-default .cm-string-2 {color: #f50;}
.cm-s-default .cm-meta {color: #555;}
.cm-s-default .cm-qualifier {color: #555;}
.cm-s-default .cm-builtin {color: #30a;}
.cm-s-default .cm-bracket {color: #997;}
.cm-s-default .cm-tag {color: #170;}
.cm-s-default .cm-attribute {color: #00c;}
.cm-s-default .cm-hr {color: #999;}
.cm-s-default .cm-link {color: #00c;}

.cm-s-default .cm-error {color: #f00;}
.cm-invalidchar {color: #f00;}

.CodeMirror-composing { border-bottom: 2px solid; }

/* Default styles for common addons */

div.CodeMirror span.CodeMirror-matchingbracket {color: #0b0;}
div.CodeMirror span.CodeMirror-nonmatchingbracket {color: #a22;}
.CodeMirror-matchingtag { background: rgba(255, 150, 0, .3); }
.CodeMirror-activeline-background {background: #e8f2ff;}

/* STOP */

/* The rest of this file contains styles related to the mechanics of
   the editor. You probably shouldn't touch them. */

.CodeMirror {
  position: relative;
  overflow: hidden;
  background: white;
}

.CodeMirror-scroll {
  overflow: scroll !important; /* Things will break if this is overridden */
  /* 50px is the magic margin used to hide the element's real scrollbars */
  /* See overflow: hidden in .CodeMirror */
  margin-bottom: -50px; margin-right: -50px;
  padding-bottom: 50px;
  height: 100%;
  outline: none; /* Prevent dragging from highlighting the element */
  position: relative;
}
.CodeMirror-sizer {
  position: relative;
  border-right: 50px solid transparent;
}

/* The fake, visible scrollbars. Used to force redraw during scrolling
   before actual scrolling happens, thus preventing shaking and
   flickering artifacts. */
.CodeMirror-vscrollbar, .CodeMirror-hscrollbar, .CodeMirror-scrollbar-filler, .CodeMirror-gutter-filler {
  position: absolute;
  z-index: 6;
  display: none;
  outline: none;
}
.CodeMirror-vscrollbar {
  right: 0; top: 0;
  overflow-x: hidden;
  overflow-y: scroll;
}
.CodeMirror-hscrollbar {
  bottom: 0; left: 0;
  overflow-y: hidden;
  overflow-x: scroll;
}
.CodeMirror-scrollbar-filler {
  right: 0; bottom: 0;
}
.CodeMirror-gutter-filler {
  left: 0; bottom: 0;
}

.CodeMirror-gutters {
  position: absolute; left: 0; top: 0;
  min-height: 100%;
  z-index: 3;
}
.CodeMirror-gutter {
  white-space: normal;
  height: 100%;
  display: inline-block;
  vertical-align: top;
  margin-bottom: -50px;
}
.CodeMirror-gutter-wrapper {
  position: absolute;
  z-index: 4;
  background: none !important;
  border: none !important;
}
.CodeMirror-gutter-background {
  position: absolute;
  top: 0; bottom: 0;
  z-index: 4;
}
.CodeMirror-gutter-elt {
  position: absolute;
  cursor: default;
  z-index: 4;
}
.CodeMirror-gutter-wrapper ::selection { background-color: transparent }
.CodeMirror-gutter-wrapper ::-moz-selection { background-color: transparent }

.CodeMirror-lines {
  cursor: text;
  min-height: 1px; /* prevents collapsing before first draw */
}
.CodeMirror pre.CodeMirror-line,
.CodeMirror pre.CodeMirror-line-like {
  /* Reset some styles that the rest of the page might have set */
  -moz-border-radius: 0; -webkit-border-radius: 0; border-radius: 0;
  border-width: 0;
  background: transparent;
  font-family: inherit;
  font-size: inherit;
  margin: 0;
  white-space: pre;
  word-wrap: normal;
  line-height: inherit;
  color: inherit;
  z-index: 2;
  position: relative;
  overflow: visible;
  -webkit-tap-highlight-color: transparent;
  -webkit-font-variant-ligatures: contextual;
  font-variant-ligatures: contextual;
}
.CodeMirror-wrap pre.CodeMirror-line,
.CodeMirror-wrap pre.CodeMirror-line-like {
  word-wrap: break-word;
  white-space: pre-wrap;
  word-break: normal;
}

.CodeMirror-linebackground {
  position: absolute;
  left: 0; right: 0; top: 0; bottom: 0;
  z-index: 0;
}

.CodeMirror-linewidget {
  position: relative;
  z-index: 2;
  padding: 0.1px; /* Force widget margins to stay inside of the container */
}

.CodeMirror-widget {}

.CodeMirror-rtl pre { direction: rtl; }

.CodeMirror-code {
  outline: none;
}

/* Force content-box sizing for the elements where we expect it */
.CodeMirror-scroll,
.CodeMirror-sizer,
.CodeMirror-gutter,
.CodeMirror-gutters,
.CodeMirror-linenumber {
  -moz-box-sizing: content-box;
  box-sizing: content-box;
}

.CodeMirror-measure {
  position: absolute;
  width: 100%;
  height: 0;
  overflow: hidden;
  visibility: hidden;
}

.CodeMirror-cursor {
  position: absolute;
  pointer-events: none;
}
.CodeMirror-measure pre { position: static; }

div.CodeMirror-cursors {
  visibility: hidden;
  position: relative;
  z-index: 3;
}
div.CodeMirror-dragcursors {
  visibility: visible;
}

.CodeMirror-focused div.CodeMirror-cursors {
  visibility: visible;
}

.CodeMirror-selected { background: #d9d9d9; }
.CodeMirror-focused .CodeMirror-selected { background: #d7d4f0; }
.CodeMirror-crosshair { cursor: crosshair; }
.CodeMirror-line::selection, .CodeMirror-line > span::selection, .CodeMirror-line > span > span::selection { background: #d7d4f0; }
.CodeMirror-line::-moz-selection, .CodeMirror-line > span::-moz-selection, .CodeMirror-line > span > span::-moz-selection { background: #d7d4f0; }

.cm-searching {
  background-color: #ffa;
  background-color: rgba(255, 255, 0, .4);
}

/* Used to force a border model for a node */
.cm-force-border { padding-right: .1px; }

@media print {
  /* Hide the cursor when printing */
  .CodeMirror div.CodeMirror-cursors {
    visibility: hidden;
  }
}

/* See issue #2901 */
.cm-tab-wrap-hack:after { content: ''; }

/* Help users use markselection to safely style text background */
span.CodeMirror-selectedtext { background: none; }

.CodeMirror-dialog {
  position: absolute;
  left: 0; right: 0;
  background: inherit;
  z-index: 15;
  padding: .1em .8em;
  overflow: hidden;
  color: inherit;
}

.CodeMirror-dialog-top {
  border-bottom: 1px solid #eee;
  top: 0;
}

.CodeMirror-dialog-bottom {
  border-top: 1px solid #eee;
  bottom: 0;
}

.CodeMirror-dialog input {
  border: none;
  outline: none;
  background: transparent;
  width: 20em;
  color: inherit;
  font-family: monospace;
}

.CodeMirror-dialog button {
  font-size: 70%;
}

.CodeMirror-foldmarker {
  color: blue;
  text-shadow: #b9f 1px 1px 2px, #b9f -1px -1px 2px, #b9f 1px -1px 2px, #b9f -1px 1px 2px;
  font-family: arial;
  line-height: .3;
  cursor: pointer;
}
.CodeMirror-foldgutter {
  width: .7em;
}
.CodeMirror-foldgutter-open,
.CodeMirror-foldgutter-folded {
  cursor: pointer;
}
.CodeMirror-foldgutter-open:after {
  content: "\25BE";
}
.CodeMirror-foldgutter-folded:after {
  content: "\25B8";
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.CodeMirror {
  line-height: var(--jp-code-line-height);
  font-size: var(--jp-code-font-size);
  font-family: var(--jp-code-font-family);
  border: 0;
  border-radius: 0;
  height: auto;
  /* Changed to auto to autogrow */
}

.CodeMirror pre {
  padding: 0 var(--jp-code-padding);
}

.jp-CodeMirrorEditor[data-type='inline'] .CodeMirror-dialog {
  background-color: var(--jp-layout-color0);
  color: var(--jp-content-font-color1);
}

/* This causes https://github.com/jupyter/jupyterlab/issues/522 */
/* May not cause it not because we changed it! */
.CodeMirror-lines {
  padding: var(--jp-code-padding) 0;
}

.CodeMirror-linenumber {
  padding: 0 8px;
}

.jp-CodeMirrorEditor {
  cursor: text;
}

.jp-CodeMirrorEditor[data-type='inline'] .CodeMirror-cursor {
  border-left: var(--jp-code-cursor-width0) solid var(--jp-editor-cursor-color);
}

/* When zoomed out 67% and 33% on a screen of 1440 width x 900 height */
@media screen and (min-width: 2138px) and (max-width: 4319px) {
  .jp-CodeMirrorEditor[data-type='inline'] .CodeMirror-cursor {
    border-left: var(--jp-code-cursor-width1) solid
      var(--jp-editor-cursor-color);
  }
}

/* When zoomed out less than 33% */
@media screen and (min-width: 4320px) {
  .jp-CodeMirrorEditor[data-type='inline'] .CodeMirror-cursor {
    border-left: var(--jp-code-cursor-width2) solid
      var(--jp-editor-cursor-color);
  }
}

.CodeMirror.jp-mod-readOnly .CodeMirror-cursor {
  display: none;
}

.CodeMirror-gutters {
  border-right: 1px solid var(--jp-border-color2);
  background-color: var(--jp-layout-color0);
}

.jp-CollaboratorCursor {
  border-left: 5px solid transparent;
  border-right: 5px solid transparent;
  border-top: none;
  border-bottom: 3px solid;
  background-clip: content-box;
  margin-left: -5px;
  margin-right: -5px;
}

.CodeMirror-selectedtext.cm-searching {
  background-color: var(--jp-search-selected-match-background-color) !important;
  color: var(--jp-search-selected-match-color) !important;
}

.cm-searching {
  background-color: var(
    --jp-search-unselected-match-background-color
  ) !important;
  color: var(--jp-search-unselected-match-color) !important;
}

.CodeMirror-focused .CodeMirror-selected {
  background-color: var(--jp-editor-selected-focused-background);
}

.CodeMirror-selected {
  background-color: var(--jp-editor-selected-background);
}

.jp-CollaboratorCursor-hover {
  position: absolute;
  z-index: 1;
  transform: translateX(-50%);
  color: white;
  border-radius: 3px;
  padding-left: 4px;
  padding-right: 4px;
  padding-top: 1px;
  padding-bottom: 1px;
  text-align: center;
  font-size: var(--jp-ui-font-size1);
  white-space: nowrap;
}

.jp-CodeMirror-ruler {
  border-left: 1px dashed var(--jp-border-color2);
}

/**
 * Here is our jupyter theme for CodeMirror syntax highlighting
 * This is used in our marked.js syntax highlighting and CodeMirror itself
 * The string "jupyter" is set in ../codemirror/widget.DEFAULT_CODEMIRROR_THEME
 * This came from the classic notebook, which came form highlight.js/GitHub
 */

/**
 * CodeMirror themes are handling the background/color in this way. This works
 * fine for CodeMirror editors outside the notebook, but the notebook styles
 * these things differently.
 */
.CodeMirror.cm-s-jupyter {
  background: var(--jp-layout-color0);
  color: var(--jp-content-font-color1);
}

/* In the notebook, we want this styling to be handled by its container */
.jp-CodeConsole .CodeMirror.cm-s-jupyter,
.jp-Notebook .CodeMirror.cm-s-jupyter {
  background: transparent;
}

.cm-s-jupyter .CodeMirror-cursor {
  border-left: var(--jp-code-cursor-width0) solid var(--jp-editor-cursor-color);
}
.cm-s-jupyter span.cm-keyword {
  color: var(--jp-mirror-editor-keyword-color);
  font-weight: bold;
}
.cm-s-jupyter span.cm-atom {
  color: var(--jp-mirror-editor-atom-color);
}
.cm-s-jupyter span.cm-number {
  color: var(--jp-mirror-editor-number-color);
}
.cm-s-jupyter span.cm-def {
  color: var(--jp-mirror-editor-def-color);
}
.cm-s-jupyter span.cm-variable {
  color: var(--jp-mirror-editor-variable-color);
}
.cm-s-jupyter span.cm-variable-2 {
  color: var(--jp-mirror-editor-variable-2-color);
}
.cm-s-jupyter span.cm-variable-3 {
  color: var(--jp-mirror-editor-variable-3-color);
}
.cm-s-jupyter span.cm-punctuation {
  color: var(--jp-mirror-editor-punctuation-color);
}
.cm-s-jupyter span.cm-property {
  color: var(--jp-mirror-editor-property-color);
}
.cm-s-jupyter span.cm-operator {
  color: var(--jp-mirror-editor-operator-color);
  font-weight: bold;
}
.cm-s-jupyter span.cm-comment {
  color: var(--jp-mirror-editor-comment-color);
  font-style: italic;
}
.cm-s-jupyter span.cm-string {
  color: var(--jp-mirror-editor-string-color);
}
.cm-s-jupyter span.cm-string-2 {
  color: var(--jp-mirror-editor-string-2-color);
}
.cm-s-jupyter span.cm-meta {
  color: var(--jp-mirror-editor-meta-color);
}
.cm-s-jupyter span.cm-qualifier {
  color: var(--jp-mirror-editor-qualifier-color);
}
.cm-s-jupyter span.cm-builtin {
  color: var(--jp-mirror-editor-builtin-color);
}
.cm-s-jupyter span.cm-bracket {
  color: var(--jp-mirror-editor-bracket-color);
}
.cm-s-jupyter span.cm-tag {
  color: var(--jp-mirror-editor-tag-color);
}
.cm-s-jupyter span.cm-attribute {
  color: var(--jp-mirror-editor-attribute-color);
}
.cm-s-jupyter span.cm-header {
  color: var(--jp-mirror-editor-header-color);
}
.cm-s-jupyter span.cm-quote {
  color: var(--jp-mirror-editor-quote-color);
}
.cm-s-jupyter span.cm-link {
  color: var(--jp-mirror-editor-link-color);
}
.cm-s-jupyter span.cm-error {
  color: var(--jp-mirror-editor-error-color);
}
.cm-s-jupyter span.cm-hr {
  color: #999;
}

.cm-s-jupyter span.cm-tab {
  background: url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADAAAAAMCAYAAAAkuj5RAAAAAXNSR0IArs4c6QAAAGFJREFUSMft1LsRQFAQheHPowAKoACx3IgEKtaEHujDjORSgWTH/ZOdnZOcM/sgk/kFFWY0qV8foQwS4MKBCS3qR6ixBJvElOobYAtivseIE120FaowJPN75GMu8j/LfMwNjh4HUpwg4LUAAAAASUVORK5CYII=);
  background-position: right;
  background-repeat: no-repeat;
}

.cm-s-jupyter .CodeMirror-activeline-background,
.cm-s-jupyter .CodeMirror-gutter {
  background-color: var(--jp-layout-color2);
}

/* Styles for shared cursors (remote cursor locations and selected ranges) */
.jp-CodeMirrorEditor .remote-caret {
  position: relative;
  border-left: 2px solid black;
  margin-left: -1px;
  margin-right: -1px;
  box-sizing: border-box;
}

.jp-CodeMirrorEditor .remote-caret > div {
  white-space: nowrap;
  position: absolute;
  top: -1.15em;
  padding-bottom: 0.05em;
  left: -2px;
  font-size: 0.95em;
  background-color: rgb(250, 129, 0);
  font-family: var(--jp-ui-font-family);
  font-weight: bold;
  line-height: normal;
  user-select: none;
  color: white;
  padding-left: 2px;
  padding-right: 2px;
  z-index: 3;
  transition: opacity 0.3s ease-in-out;
}

.jp-CodeMirrorEditor .remote-caret.hide-name > div {
  transition-delay: 0.7s;
  opacity: 0;
}

.jp-CodeMirrorEditor .remote-caret:hover > div {
  opacity: 1;
  transition-delay: 0s;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| RenderedText
|----------------------------------------------------------------------------*/

:root {
  /* This is the padding value to fill the gaps between lines containing spans with background color. */
  --jp-private-code-span-padding: calc(
    (var(--jp-code-line-height) - 1) * var(--jp-code-font-size) / 2
  );
}

.jp-RenderedText {
  text-align: left;
  padding-left: var(--jp-code-padding);
  line-height: var(--jp-code-line-height);
  font-family: var(--jp-code-font-family);
}

.jp-RenderedText pre,
.jp-RenderedJavaScript pre,
.jp-RenderedHTMLCommon pre {
  color: var(--jp-content-font-color1);
  font-size: var(--jp-code-font-size);
  border: none;
  margin: 0px;
  padding: 0px;
}

.jp-RenderedText pre a:link {
  text-decoration: none;
  color: var(--jp-content-link-color);
}
.jp-RenderedText pre a:hover {
  text-decoration: underline;
  color: var(--jp-content-link-color);
}
.jp-RenderedText pre a:visited {
  text-decoration: none;
  color: var(--jp-content-link-color);
}

/* console foregrounds and backgrounds */
.jp-RenderedText pre .ansi-black-fg {
  color: #3e424d;
}
.jp-RenderedText pre .ansi-red-fg {
  color: #e75c58;
}
.jp-RenderedText pre .ansi-green-fg {
  color: #00a250;
}
.jp-RenderedText pre .ansi-yellow-fg {
  color: #ddb62b;
}
.jp-RenderedText pre .ansi-blue-fg {
  color: #208ffb;
}
.jp-RenderedText pre .ansi-magenta-fg {
  color: #d160c4;
}
.jp-RenderedText pre .ansi-cyan-fg {
  color: #60c6c8;
}
.jp-RenderedText pre .ansi-white-fg {
  color: #c5c1b4;
}

.jp-RenderedText pre .ansi-black-bg {
  background-color: #3e424d;
  padding: var(--jp-private-code-span-padding) 0;
}
.jp-RenderedText pre .ansi-red-bg {
  background-color: #e75c58;
  padding: var(--jp-private-code-span-padding) 0;
}
.jp-RenderedText pre .ansi-green-bg {
  background-color: #00a250;
  padding: var(--jp-private-code-span-padding) 0;
}
.jp-RenderedText pre .ansi-yellow-bg {
  background-color: #ddb62b;
  padding: var(--jp-private-code-span-padding) 0;
}
.jp-RenderedText pre .ansi-blue-bg {
  background-color: #208ffb;
  padding: var(--jp-private-code-span-padding) 0;
}
.jp-RenderedText pre .ansi-magenta-bg {
  background-color: #d160c4;
  padding: var(--jp-private-code-span-padding) 0;
}
.jp-RenderedText pre .ansi-cyan-bg {
  background-color: #60c6c8;
  padding: var(--jp-private-code-span-padding) 0;
}
.jp-RenderedText pre .ansi-white-bg {
  background-color: #c5c1b4;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-black-intense-fg {
  color: #282c36;
}
.jp-RenderedText pre .ansi-red-intense-fg {
  color: #b22b31;
}
.jp-RenderedText pre .ansi-green-intense-fg {
  color: #007427;
}
.jp-RenderedText pre .ansi-yellow-intense-fg {
  color: #b27d12;
}
.jp-RenderedText pre .ansi-blue-intense-fg {
  color: #0065ca;
}
.jp-RenderedText pre .ansi-magenta-intense-fg {
  color: #a03196;
}
.jp-RenderedText pre .ansi-cyan-intense-fg {
  color: #258f8f;
}
.jp-RenderedText pre .ansi-white-intense-fg {
  color: #a1a6b2;
}

.jp-RenderedText pre .ansi-black-intense-bg {
  background-color: #282c36;
  padding: var(--jp-private-code-span-padding) 0;
}
.jp-RenderedText pre .ansi-red-intense-bg {
  background-color: #b22b31;
  padding: var(--jp-private-code-span-padding) 0;
}
.jp-RenderedText pre .ansi-green-intense-bg {
  background-color: #007427;
  padding: var(--jp-private-code-span-padding) 0;
}
.jp-RenderedText pre .ansi-yellow-intense-bg {
  background-color: #b27d12;
  padding: var(--jp-private-code-span-padding) 0;
}
.jp-RenderedText pre .ansi-blue-intense-bg {
  background-color: #0065ca;
  padding: var(--jp-private-code-span-padding) 0;
}
.jp-RenderedText pre .ansi-magenta-intense-bg {
  background-color: #a03196;
  padding: var(--jp-private-code-span-padding) 0;
}
.jp-RenderedText pre .ansi-cyan-intense-bg {
  background-color: #258f8f;
  padding: var(--jp-private-code-span-padding) 0;
}
.jp-RenderedText pre .ansi-white-intense-bg {
  background-color: #a1a6b2;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-default-inverse-fg {
  color: var(--jp-ui-inverse-font-color0);
}
.jp-RenderedText pre .ansi-default-inverse-bg {
  background-color: var(--jp-inverse-layout-color0);
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-bold {
  font-weight: bold;
}
.jp-RenderedText pre .ansi-underline {
  text-decoration: underline;
}

.jp-RenderedText[data-mime-type='application/vnd.jupyter.stderr'] {
  background: var(--jp-rendermime-error-background);
  padding-top: var(--jp-code-padding);
}

/*-----------------------------------------------------------------------------
| RenderedLatex
|----------------------------------------------------------------------------*/

.jp-RenderedLatex {
  color: var(--jp-content-font-color1);
  font-size: var(--jp-content-font-size1);
  line-height: var(--jp-content-line-height);
}

/* Left-justify outputs.*/
.jp-OutputArea-output.jp-RenderedLatex {
  padding: var(--jp-code-padding);
  text-align: left;
}

/*-----------------------------------------------------------------------------
| RenderedHTML
|----------------------------------------------------------------------------*/

.jp-RenderedHTMLCommon {
  color: var(--jp-content-font-color1);
  font-family: var(--jp-content-font-family);
  font-size: var(--jp-content-font-size1);
  line-height: var(--jp-content-line-height);
  /* Give a bit more R padding on Markdown text to keep line lengths reasonable */
  padding-right: 20px;
}

.jp-RenderedHTMLCommon em {
  font-style: italic;
}

.jp-RenderedHTMLCommon strong {
  font-weight: bold;
}

.jp-RenderedHTMLCommon u {
  text-decoration: underline;
}

.jp-RenderedHTMLCommon a:link {
  text-decoration: none;
  color: var(--jp-content-link-color);
}

.jp-RenderedHTMLCommon a:hover {
  text-decoration: underline;
  color: var(--jp-content-link-color);
}

.jp-RenderedHTMLCommon a:visited {
  text-decoration: none;
  color: var(--jp-content-link-color);
}

/* Headings */

.jp-RenderedHTMLCommon h1,
.jp-RenderedHTMLCommon h2,
.jp-RenderedHTMLCommon h3,
.jp-RenderedHTMLCommon h4,
.jp-RenderedHTMLCommon h5,
.jp-RenderedHTMLCommon h6 {
  line-height: var(--jp-content-heading-line-height);
  font-weight: var(--jp-content-heading-font-weight);
  font-style: normal;
  margin: var(--jp-content-heading-margin-top) 0
    var(--jp-content-heading-margin-bottom) 0;
}

.jp-RenderedHTMLCommon h1:first-child,
.jp-RenderedHTMLCommon h2:first-child,
.jp-RenderedHTMLCommon h3:first-child,
.jp-RenderedHTMLCommon h4:first-child,
.jp-RenderedHTMLCommon h5:first-child,
.jp-RenderedHTMLCommon h6:first-child {
  margin-top: calc(0.5 * var(--jp-content-heading-margin-top));
}

.jp-RenderedHTMLCommon h1:last-child,
.jp-RenderedHTMLCommon h2:last-child,
.jp-RenderedHTMLCommon h3:last-child,
.jp-RenderedHTMLCommon h4:last-child,
.jp-RenderedHTMLCommon h5:last-child,
.jp-RenderedHTMLCommon h6:last-child {
  margin-bottom: calc(0.5 * var(--jp-content-heading-margin-bottom));
}

.jp-RenderedHTMLCommon h1 {
  font-size: var(--jp-content-font-size5);
}

.jp-RenderedHTMLCommon h2 {
  font-size: var(--jp-content-font-size4);
}

.jp-RenderedHTMLCommon h3 {
  font-size: var(--jp-content-font-size3);
}

.jp-RenderedHTMLCommon h4 {
  font-size: var(--jp-content-font-size2);
}

.jp-RenderedHTMLCommon h5 {
  font-size: var(--jp-content-font-size1);
}

.jp-RenderedHTMLCommon h6 {
  font-size: var(--jp-content-font-size0);
}

/* Lists */

.jp-RenderedHTMLCommon ul:not(.list-inline),
.jp-RenderedHTMLCommon ol:not(.list-inline) {
  padding-left: 2em;
}

.jp-RenderedHTMLCommon ul {
  list-style: disc;
}

.jp-RenderedHTMLCommon ul ul {
  list-style: square;
}

.jp-RenderedHTMLCommon ul ul ul {
  list-style: circle;
}

.jp-RenderedHTMLCommon ol {
  list-style: decimal;
}

.jp-RenderedHTMLCommon ol ol {
  list-style: upper-alpha;
}

.jp-RenderedHTMLCommon ol ol ol {
  list-style: lower-alpha;
}

.jp-RenderedHTMLCommon ol ol ol ol {
  list-style: lower-roman;
}

.jp-RenderedHTMLCommon ol ol ol ol ol {
  list-style: decimal;
}

.jp-RenderedHTMLCommon ol,
.jp-RenderedHTMLCommon ul {
  margin-bottom: 1em;
}

.jp-RenderedHTMLCommon ul ul,
.jp-RenderedHTMLCommon ul ol,
.jp-RenderedHTMLCommon ol ul,
.jp-RenderedHTMLCommon ol ol {
  margin-bottom: 0em;
}

.jp-RenderedHTMLCommon hr {
  color: var(--jp-border-color2);
  background-color: var(--jp-border-color1);
  margin-top: 1em;
  margin-bottom: 1em;
}

.jp-RenderedHTMLCommon > pre {
  margin: 1.5em 2em;
}

.jp-RenderedHTMLCommon pre,
.jp-RenderedHTMLCommon code {
  border: 0;
  background-color: var(--jp-layout-color0);
  color: var(--jp-content-font-color1);
  font-family: var(--jp-code-font-family);
  font-size: inherit;
  line-height: var(--jp-code-line-height);
  padding: 0;
  white-space: pre-wrap;
}

.jp-RenderedHTMLCommon :not(pre) > code {
  background-color: var(--jp-layout-color2);
  padding: 1px 5px;
}

/* Tables */

.jp-RenderedHTMLCommon table {
  border-collapse: collapse;
  border-spacing: 0;
  border: none;
  color: var(--jp-ui-font-color1);
  font-size: 12px;
  table-layout: fixed;
  margin-left: auto;
  margin-right: auto;
}

.jp-RenderedHTMLCommon thead {
  border-bottom: var(--jp-border-width) solid var(--jp-border-color1);
  vertical-align: bottom;
}

.jp-RenderedHTMLCommon td,
.jp-RenderedHTMLCommon th,
.jp-RenderedHTMLCommon tr {
  vertical-align: middle;
  padding: 0.5em 0.5em;
  line-height: normal;
  white-space: normal;
  max-width: none;
  border: none;
}

.jp-RenderedMarkdown.jp-RenderedHTMLCommon td,
.jp-RenderedMarkdown.jp-RenderedHTMLCommon th {
  max-width: none;
}

:not(.jp-RenderedMarkdown).jp-RenderedHTMLCommon td,
:not(.jp-RenderedMarkdown).jp-RenderedHTMLCommon th,
:not(.jp-RenderedMarkdown).jp-RenderedHTMLCommon tr {
  text-align: right;
}

.jp-RenderedHTMLCommon th {
  font-weight: bold;
}

.jp-RenderedHTMLCommon tbody tr:nth-child(odd) {
  background: var(--jp-layout-color0);
}

.jp-RenderedHTMLCommon tbody tr:nth-child(even) {
  background: var(--jp-rendermime-table-row-background);
}

.jp-RenderedHTMLCommon tbody tr:hover {
  background: var(--jp-rendermime-table-row-hover-background);
}

.jp-RenderedHTMLCommon table {
  margin-bottom: 1em;
}

.jp-RenderedHTMLCommon p {
  text-align: left;
  margin: 0px;
}

.jp-RenderedHTMLCommon p {
  margin-bottom: 1em;
}

.jp-RenderedHTMLCommon img {
  -moz-force-broken-image-icon: 1;
}

/* Restrict to direct children as other images could be nested in other content. */
.jp-RenderedHTMLCommon > img {
  display: block;
  margin-left: 0;
  margin-right: 0;
  margin-bottom: 1em;
}

/* Change color behind transparent images if they need it... */
[data-jp-theme-light='false'] .jp-RenderedImage img.jp-needs-light-background {
  background-color: var(--jp-inverse-layout-color1);
}
[data-jp-theme-light='true'] .jp-RenderedImage img.jp-needs-dark-background {
  background-color: var(--jp-inverse-layout-color1);
}
/* ...or leave it untouched if they don't */
[data-jp-theme-light='false'] .jp-RenderedImage img.jp-needs-dark-background {
}
[data-jp-theme-light='true'] .jp-RenderedImage img.jp-needs-light-background {
}

.jp-RenderedHTMLCommon img,
.jp-RenderedImage img,
.jp-RenderedHTMLCommon svg,
.jp-RenderedSVG svg {
  max-width: 100%;
  height: auto;
}

.jp-RenderedHTMLCommon img.jp-mod-unconfined,
.jp-RenderedImage img.jp-mod-unconfined,
.jp-RenderedHTMLCommon svg.jp-mod-unconfined,
.jp-RenderedSVG svg.jp-mod-unconfined {
  max-width: none;
}

.jp-RenderedHTMLCommon .alert {
  padding: var(--jp-notebook-padding);
  border: var(--jp-border-width) solid transparent;
  border-radius: var(--jp-border-radius);
  margin-bottom: 1em;
}

.jp-RenderedHTMLCommon .alert-info {
  color: var(--jp-info-color0);
  background-color: var(--jp-info-color3);
  border-color: var(--jp-info-color2);
}
.jp-RenderedHTMLCommon .alert-info hr {
  border-color: var(--jp-info-color3);
}
.jp-RenderedHTMLCommon .alert-info > p:last-child,
.jp-RenderedHTMLCommon .alert-info > ul:last-child {
  margin-bottom: 0;
}

.jp-RenderedHTMLCommon .alert-warning {
  color: var(--jp-warn-color0);
  background-color: var(--jp-warn-color3);
  border-color: var(--jp-warn-color2);
}
.jp-RenderedHTMLCommon .alert-warning hr {
  border-color: var(--jp-warn-color3);
}
.jp-RenderedHTMLCommon .alert-warning > p:last-child,
.jp-RenderedHTMLCommon .alert-warning > ul:last-child {
  margin-bottom: 0;
}

.jp-RenderedHTMLCommon .alert-success {
  color: var(--jp-success-color0);
  background-color: var(--jp-success-color3);
  border-color: var(--jp-success-color2);
}
.jp-RenderedHTMLCommon .alert-success hr {
  border-color: var(--jp-success-color3);
}
.jp-RenderedHTMLCommon .alert-success > p:last-child,
.jp-RenderedHTMLCommon .alert-success > ul:last-child {
  margin-bottom: 0;
}

.jp-RenderedHTMLCommon .alert-danger {
  color: var(--jp-error-color0);
  background-color: var(--jp-error-color3);
  border-color: var(--jp-error-color2);
}
.jp-RenderedHTMLCommon .alert-danger hr {
  border-color: var(--jp-error-color3);
}
.jp-RenderedHTMLCommon .alert-danger > p:last-child,
.jp-RenderedHTMLCommon .alert-danger > ul:last-child {
  margin-bottom: 0;
}

.jp-RenderedHTMLCommon blockquote {
  margin: 1em 2em;
  padding: 0 1em;
  border-left: 5px solid var(--jp-border-color2);
}

a.jp-InternalAnchorLink {
  visibility: hidden;
  margin-left: 8px;
  color: var(--md-blue-800);
}

h1:hover .jp-InternalAnchorLink,
h2:hover .jp-InternalAnchorLink,
h3:hover .jp-InternalAnchorLink,
h4:hover .jp-InternalAnchorLink,
h5:hover .jp-InternalAnchorLink,
h6:hover .jp-InternalAnchorLink {
  visibility: visible;
}

.jp-RenderedHTMLCommon kbd {
  background-color: var(--jp-rendermime-table-row-background);
  border: 1px solid var(--jp-border-color0);
  border-bottom-color: var(--jp-border-color2);
  border-radius: 3px;
  box-shadow: inset 0 -1px 0 rgba(0, 0, 0, 0.25);
  display: inline-block;
  font-size: 0.8em;
  line-height: 1em;
  padding: 0.2em 0.5em;
}

/* Most direct children of .jp-RenderedHTMLCommon have a margin-bottom of 1.0.
 * At the bottom of cells this is a bit too much as there is also spacing
 * between cells. Going all the way to 0 gets too tight between markdown and
 * code cells.
 */
.jp-RenderedHTMLCommon > *:last-child {
  margin-bottom: 0.5em;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-MimeDocument {
  outline: none;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Variables
|----------------------------------------------------------------------------*/

:root {
  --jp-private-filebrowser-button-height: 28px;
  --jp-private-filebrowser-button-width: 48px;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-FileBrowser {
  display: flex;
  flex-direction: column;
  color: var(--jp-ui-font-color1);
  background: var(--jp-layout-color1);
  /* This is needed so that all font sizing of children done in ems is
   * relative to this base size */
  font-size: var(--jp-ui-font-size1);
}

.jp-FileBrowser-toolbar.jp-Toolbar {
  border-bottom: none;
  height: auto;
  margin: var(--jp-toolbar-header-margin);
  box-shadow: none;
}

.jp-BreadCrumbs {
  flex: 0 0 auto;
  margin: 8px 12px 8px 12px;
}

.jp-BreadCrumbs-item {
  margin: 0px 2px;
  padding: 0px 2px;
  border-radius: var(--jp-border-radius);
  cursor: pointer;
}

.jp-BreadCrumbs-item:hover {
  background-color: var(--jp-layout-color2);
}

.jp-BreadCrumbs-item:first-child {
  margin-left: 0px;
}

.jp-BreadCrumbs-item.jp-mod-dropTarget {
  background-color: var(--jp-brand-color2);
  opacity: 0.7;
}

/*-----------------------------------------------------------------------------
| Buttons
|----------------------------------------------------------------------------*/

.jp-FileBrowser-toolbar.jp-Toolbar {
  padding: 0px;
  margin: 8px 12px 0px 12px;
}

.jp-FileBrowser-toolbar.jp-Toolbar {
  justify-content: flex-start;
}

.jp-FileBrowser-toolbar.jp-Toolbar .jp-Toolbar-item {
  flex: 0 0 auto;
  padding-left: 0px;
  padding-right: 2px;
}

.jp-FileBrowser-toolbar.jp-Toolbar .jp-ToolbarButtonComponent {
  width: 40px;
}

.jp-FileBrowser-toolbar.jp-Toolbar
  .jp-Toolbar-item:first-child
  .jp-ToolbarButtonComponent {
  width: 72px;
  background: var(--jp-brand-color1);
}

.jp-FileBrowser-toolbar.jp-Toolbar
  .jp-Toolbar-item:first-child
  .jp-ToolbarButtonComponent:focus-visible {
  background-color: var(--jp-brand-color0);
}

.jp-FileBrowser-toolbar.jp-Toolbar
  .jp-Toolbar-item:first-child
  .jp-ToolbarButtonComponent
  .jp-icon3 {
  fill: white;
}

/*-----------------------------------------------------------------------------
| Other styles
|----------------------------------------------------------------------------*/

.jp-FileDialog.jp-mod-conflict input {
  color: var(--jp-error-color1);
}

.jp-FileDialog .jp-new-name-title {
  margin-top: 12px;
}

.jp-LastModified-hidden {
  display: none;
}

.jp-FileBrowser-filterBox {
  padding: 0px;
  flex: 0 0 auto;
  margin: 8px 12px 0px 12px;
}

/*-----------------------------------------------------------------------------
| DirListing
|----------------------------------------------------------------------------*/

.jp-DirListing {
  flex: 1 1 auto;
  display: flex;
  flex-direction: column;
  outline: 0;
}

.jp-DirListing:focus-visible {
  border: 1px solid var(--jp-brand-color1);
}

.jp-DirListing-header {
  flex: 0 0 auto;
  display: flex;
  flex-direction: row;
  overflow: hidden;
  border-top: var(--jp-border-width) solid var(--jp-border-color2);
  border-bottom: var(--jp-border-width) solid var(--jp-border-color1);
  box-shadow: var(--jp-toolbar-box-shadow);
  z-index: 2;
}

.jp-DirListing-headerItem {
  padding: 4px 12px 2px 12px;
  font-weight: 500;
}

.jp-DirListing-headerItem:hover {
  background: var(--jp-layout-color2);
}

.jp-DirListing-headerItem.jp-id-name {
  flex: 1 0 84px;
}

.jp-DirListing-headerItem.jp-id-modified {
  flex: 0 0 112px;
  border-left: var(--jp-border-width) solid var(--jp-border-color2);
  text-align: right;
}

.jp-id-narrow {
  display: none;
  flex: 0 0 5px;
  padding: 4px 4px;
  border-left: var(--jp-border-width) solid var(--jp-border-color2);
  text-align: right;
  color: var(--jp-border-color2);
}

.jp-DirListing-narrow .jp-id-narrow {
  display: block;
}

.jp-DirListing-narrow .jp-id-modified,
.jp-DirListing-narrow .jp-DirListing-itemModified {
  display: none;
}

.jp-DirListing-headerItem.jp-mod-selected {
  font-weight: 600;
}

/* increase specificity to override bundled default */
.jp-DirListing-content {
  flex: 1 1 auto;
  margin: 0;
  padding: 0;
  list-style-type: none;
  overflow: auto;
  background-color: var(--jp-layout-color1);
}

.jp-DirListing-content mark {
  color: var(--jp-ui-font-color0);
  background-color: transparent;
  font-weight: bold;
}

.jp-DirListing-content .jp-DirListing-item.jp-mod-selected mark {
  color: var(--jp-ui-inverse-font-color0);
}

/* Style the directory listing content when a user drops a file to upload */
.jp-DirListing.jp-mod-native-drop .jp-DirListing-content {
  outline: 5px dashed rgba(128, 128, 128, 0.5);
  outline-offset: -10px;
  cursor: copy;
}

.jp-DirListing-item {
  display: flex;
  flex-direction: row;
  padding: 4px 12px;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}

.jp-DirListing-item[data-is-dot] {
  opacity: 75%;
}

.jp-DirListing-item.jp-mod-selected {
  color: var(--jp-ui-inverse-font-color1);
  background: var(--jp-brand-color1);
}

.jp-DirListing-item.jp-mod-dropTarget {
  background: var(--jp-brand-color3);
}

.jp-DirListing-item:hover:not(.jp-mod-selected) {
  background: var(--jp-layout-color2);
}

.jp-DirListing-itemIcon {
  flex: 0 0 20px;
  margin-right: 4px;
}

.jp-DirListing-itemText {
  flex: 1 0 64px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  user-select: none;
}

.jp-DirListing-itemModified {
  flex: 0 0 125px;
  text-align: right;
}

.jp-DirListing-editor {
  flex: 1 0 64px;
  outline: none;
  border: none;
}

.jp-DirListing-item.jp-mod-running .jp-DirListing-itemIcon:before {
  color: var(--jp-success-color1);
  content: '\25CF';
  font-size: 8px;
  position: absolute;
  left: -8px;
}

.jp-DirListing-item.jp-mod-running.jp-mod-selected
  .jp-DirListing-itemIcon:before {
  color: var(--jp-ui-inverse-font-color1);
}

.jp-DirListing-item.lm-mod-drag-image,
.jp-DirListing-item.jp-mod-selected.lm-mod-drag-image {
  font-size: var(--jp-ui-font-size1);
  padding-left: 4px;
  margin-left: 4px;
  width: 160px;
  background-color: var(--jp-ui-inverse-font-color2);
  box-shadow: var(--jp-elevation-z2);
  border-radius: 0px;
  color: var(--jp-ui-font-color1);
  transform: translateX(-40%) translateY(-58%);
}

.jp-DirListing-deadSpace {
  flex: 1 1 auto;
  margin: 0;
  padding: 0;
  list-style-type: none;
  overflow: auto;
  background-color: var(--jp-layout-color1);
}

.jp-Document {
  min-width: 120px;
  min-height: 120px;
  outline: none;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Private CSS variables
|----------------------------------------------------------------------------*/

:root {
}

/*-----------------------------------------------------------------------------
| Main OutputArea
| OutputArea has a list of Outputs
|----------------------------------------------------------------------------*/

.jp-OutputArea {
  overflow-y: auto;
}

.jp-OutputArea-child {
  display: flex;
  flex-direction: row;
}

body[data-format='mobile'] .jp-OutputArea-child {
  flex-direction: column;
}

.jp-OutputPrompt {
  flex: 0 0 var(--jp-cell-prompt-width);
  color: var(--jp-cell-outprompt-font-color);
  font-family: var(--jp-cell-prompt-font-family);
  padding: var(--jp-code-padding);
  letter-spacing: var(--jp-cell-prompt-letter-spacing);
  line-height: var(--jp-code-line-height);
  font-size: var(--jp-code-font-size);
  border: var(--jp-border-width) solid transparent;
  opacity: var(--jp-cell-prompt-opacity);
  /* Right align prompt text, don't wrap to handle large prompt numbers */
  text-align: right;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  /* Disable text selection */
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}

body[data-format='mobile'] .jp-OutputPrompt {
  flex: 0 0 auto;
  text-align: left;
}

.jp-OutputArea-output {
  height: auto;
  overflow: auto;
  user-select: text;
  -moz-user-select: text;
  -webkit-user-select: text;
  -ms-user-select: text;
}

.jp-OutputArea-child .jp-OutputArea-output {
  flex-grow: 1;
  flex-shrink: 1;
}

body[data-format='mobile'] .jp-OutputArea-child .jp-OutputArea-output {
  margin-left: var(--jp-notebook-padding);
}

/**
 * Isolated output.
 */
.jp-OutputArea-output.jp-mod-isolated {
  width: 100%;
  display: block;
}

/*
When drag events occur, `p-mod-override-cursor` is added to the body.
Because iframes steal all cursor events, the following two rules are necessary
to suppress pointer events while resize drags are occurring. There may be a
better solution to this problem.
*/
body.lm-mod-override-cursor .jp-OutputArea-output.jp-mod-isolated {
  position: relative;
}

body.lm-mod-override-cursor .jp-OutputArea-output.jp-mod-isolated:before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: transparent;
}

/* pre */

.jp-OutputArea-output pre {
  border: none;
  margin: 0px;
  padding: 0px;
  overflow-x: auto;
  overflow-y: auto;
  word-break: break-all;
  word-wrap: break-word;
  white-space: pre-wrap;
}

/* tables */

.jp-OutputArea-output.jp-RenderedHTMLCommon table {
  margin-left: 0;
  margin-right: 0;
}

/* description lists */

.jp-OutputArea-output dl,
.jp-OutputArea-output dt,
.jp-OutputArea-output dd {
  display: block;
}

.jp-OutputArea-output dl {
  width: 100%;
  overflow: hidden;
  padding: 0;
  margin: 0;
}

.jp-OutputArea-output dt {
  font-weight: bold;
  float: left;
  width: 20%;
  padding: 0;
  margin: 0;
}

.jp-OutputArea-output dd {
  float: left;
  width: 80%;
  padding: 0;
  margin: 0;
}

/* Hide the gutter in case of
 *  - nested output areas (e.g. in the case of output widgets)
 *  - mirrored output areas
 */
.jp-OutputArea .jp-OutputArea .jp-OutputArea-prompt {
  display: none;
}

/*-----------------------------------------------------------------------------
| executeResult is added to any Output-result for the display of the object
| returned by a cell
|----------------------------------------------------------------------------*/

.jp-OutputArea-output.jp-OutputArea-executeResult {
  margin-left: 0px;
  flex: 1 1 auto;
}

/* Text output with the Out[] prompt needs a top padding to match the
 * alignment of the Out[] prompt itself.
 */
.jp-OutputArea-executeResult .jp-RenderedText.jp-OutputArea-output {
  padding-top: var(--jp-code-padding);
  border-top: var(--jp-border-width) solid transparent;
}

/*-----------------------------------------------------------------------------
| The Stdin output
|----------------------------------------------------------------------------*/

.jp-OutputArea-stdin {
  line-height: var(--jp-code-line-height);
  padding-top: var(--jp-code-padding);
  display: flex;
}

.jp-Stdin-prompt {
  color: var(--jp-content-font-color0);
  padding-right: var(--jp-code-padding);
  vertical-align: baseline;
  flex: 0 0 auto;
}

.jp-Stdin-input {
  font-family: var(--jp-code-font-family);
  font-size: inherit;
  color: inherit;
  background-color: inherit;
  width: 42%;
  min-width: 200px;
  /* make sure input baseline aligns with prompt */
  vertical-align: baseline;
  /* padding + margin = 0.5em between prompt and cursor */
  padding: 0em 0.25em;
  margin: 0em 0.25em;
  flex: 0 0 70%;
}

.jp-Stdin-input:focus {
  box-shadow: none;
}

/*-----------------------------------------------------------------------------
| Output Area View
|----------------------------------------------------------------------------*/

.jp-LinkedOutputView .jp-OutputArea {
  height: 100%;
  display: block;
}

.jp-LinkedOutputView .jp-OutputArea-output:only-child {
  height: 100%;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-Collapser {
  flex: 0 0 var(--jp-cell-collapser-width);
  padding: 0px;
  margin: 0px;
  border: none;
  outline: none;
  background: transparent;
  border-radius: var(--jp-border-radius);
  opacity: 1;
}

.jp-Collapser-child {
  display: block;
  width: 100%;
  box-sizing: border-box;
  /* height: 100% doesn't work because the height of its parent is computed from content */
  position: absolute;
  top: 0px;
  bottom: 0px;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Header/Footer
|----------------------------------------------------------------------------*/

/* Hidden by zero height by default */
.jp-CellHeader,
.jp-CellFooter {
  height: 0px;
  width: 100%;
  padding: 0px;
  margin: 0px;
  border: none;
  outline: none;
  background: transparent;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Input
|----------------------------------------------------------------------------*/

/* All input areas */
.jp-InputArea {
  display: flex;
  flex-direction: row;
  overflow: hidden;
}

body[data-format='mobile'] .jp-InputArea {
  flex-direction: column;
}

.jp-InputArea-editor {
  flex: 1 1 auto;
  overflow: hidden;
}

.jp-InputArea-editor {
  /* This is the non-active, default styling */
  border: var(--jp-border-width) solid var(--jp-cell-editor-border-color);
  border-radius: 0px;
  background: var(--jp-cell-editor-background);
}

body[data-format='mobile'] .jp-InputArea-editor {
  margin-left: var(--jp-notebook-padding);
}

.jp-InputPrompt {
  flex: 0 0 var(--jp-cell-prompt-width);
  color: var(--jp-cell-inprompt-font-color);
  font-family: var(--jp-cell-prompt-font-family);
  padding: var(--jp-code-padding);
  letter-spacing: var(--jp-cell-prompt-letter-spacing);
  opacity: var(--jp-cell-prompt-opacity);
  line-height: var(--jp-code-line-height);
  font-size: var(--jp-code-font-size);
  border: var(--jp-border-width) solid transparent;
  opacity: var(--jp-cell-prompt-opacity);
  /* Right align prompt text, don't wrap to handle large prompt numbers */
  text-align: right;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  /* Disable text selection */
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}

body[data-format='mobile'] .jp-InputPrompt {
  flex: 0 0 auto;
  text-align: left;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Placeholder
|----------------------------------------------------------------------------*/

.jp-Placeholder {
  display: flex;
  flex-direction: row;
  flex: 1 1 auto;
}

.jp-Placeholder-prompt {
  box-sizing: border-box;
}

.jp-Placeholder-content {
  flex: 1 1 auto;
  border: none;
  background: transparent;
  height: 20px;
  box-sizing: border-box;
}

.jp-Placeholder-content .jp-MoreHorizIcon {
  width: 32px;
  height: 16px;
  border: 1px solid transparent;
  border-radius: var(--jp-border-radius);
}

.jp-Placeholder-content .jp-MoreHorizIcon:hover {
  border: 1px solid var(--jp-border-color1);
  box-shadow: 0px 0px 2px 0px rgba(0, 0, 0, 0.25);
  background-color: var(--jp-layout-color0);
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Private CSS variables
|----------------------------------------------------------------------------*/

:root {
  --jp-private-cell-scrolling-output-offset: 5px;
}

/*-----------------------------------------------------------------------------
| Cell
|----------------------------------------------------------------------------*/

.jp-Cell {
  padding: var(--jp-cell-padding);
  margin: 0px;
  border: none;
  outline: none;
  background: transparent;
}

/*-----------------------------------------------------------------------------
| Common input/output
|----------------------------------------------------------------------------*/

.jp-Cell-inputWrapper,
.jp-Cell-outputWrapper {
  display: flex;
  flex-direction: row;
  padding: 0px;
  margin: 0px;
  /* Added to reveal the box-shadow on the input and output collapsers. */
  overflow: visible;
}

/* Only input/output areas inside cells */
.jp-Cell-inputArea,
.jp-Cell-outputArea {
  flex: 1 1 auto;
}

/*-----------------------------------------------------------------------------
| Collapser
|----------------------------------------------------------------------------*/

/* Make the output collapser disappear when there is not output, but do so
 * in a manner that leaves it in the layout and preserves its width.
 */
.jp-Cell.jp-mod-noOutputs .jp-Cell-outputCollapser {
  border: none !important;
  background: transparent !important;
}

.jp-Cell:not(.jp-mod-noOutputs) .jp-Cell-outputCollapser {
  min-height: var(--jp-cell-collapser-min-height);
}

/*-----------------------------------------------------------------------------
| Output
|----------------------------------------------------------------------------*/

/* Put a space between input and output when there IS output */
.jp-Cell:not(.jp-mod-noOutputs) .jp-Cell-outputWrapper {
  margin-top: 5px;
}

.jp-CodeCell.jp-mod-outputsScrolled .jp-Cell-outputArea {
  overflow-y: auto;
  max-height: 200px;
  box-shadow: inset 0 0 6px 2px rgba(0, 0, 0, 0.3);
  margin-left: var(--jp-private-cell-scrolling-output-offset);
}

.jp-CodeCell.jp-mod-outputsScrolled .jp-OutputArea-prompt {
  flex: 0 0
    calc(
      var(--jp-cell-prompt-width) -
        var(--jp-private-cell-scrolling-output-offset)
    );
}

/*-----------------------------------------------------------------------------
| CodeCell
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| MarkdownCell
|----------------------------------------------------------------------------*/

.jp-MarkdownOutput {
  flex: 1 1 auto;
  margin-top: 0;
  margin-bottom: 0;
  padding-left: var(--jp-code-padding);
}

.jp-MarkdownOutput.jp-RenderedHTMLCommon {
  overflow: auto;
}

.jp-showHiddenCellsButton {
  margin-left: calc(var(--jp-cell-prompt-width) + 2 * var(--jp-code-padding));
  margin-top: var(--jp-code-padding);
  border: 1px solid var(--jp-border-color2);
  background-color: var(--jp-border-color3) !important;
  color: var(--jp-content-font-color0) !important;
}

.jp-showHiddenCellsButton:hover {
  background-color: var(--jp-border-color2) !important;
}

.jp-collapseHeadingButton {
  display: none;
}

.jp-MarkdownCell:hover .jp-collapseHeadingButton {
  display: flex;
  min-height: var(--jp-cell-collapser-min-height);
  position: absolute;
  right: 0;
  top: 0;
  bottom: 0;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Variables
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------

/*-----------------------------------------------------------------------------
| Styles
|----------------------------------------------------------------------------*/

.jp-NotebookPanel-toolbar {
  padding: 2px;
}

.jp-Toolbar-item.jp-Notebook-toolbarCellType .jp-select-wrapper.jp-mod-focused {
  border: none;
  box-shadow: none;
}

.jp-Notebook-toolbarCellTypeDropdown select {
  height: 24px;
  font-size: var(--jp-ui-font-size1);
  line-height: 14px;
  border-radius: 0;
  display: block;
}

.jp-Notebook-toolbarCellTypeDropdown span {
  top: 5px !important;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Private CSS variables
|----------------------------------------------------------------------------*/

:root {
  --jp-private-notebook-dragImage-width: 304px;
  --jp-private-notebook-dragImage-height: 36px;
  --jp-private-notebook-selected-color: var(--md-blue-400);
  --jp-private-notebook-active-color: var(--md-green-400);
}

/*-----------------------------------------------------------------------------
| Imports
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Notebook
|----------------------------------------------------------------------------*/

.jp-NotebookPanel {
  display: block;
  height: 100%;
}

.jp-NotebookPanel.jp-Document {
  min-width: 240px;
  min-height: 120px;
}

.jp-Notebook {
  padding: var(--jp-notebook-padding);
  outline: none;
  overflow: auto;
  background: var(--jp-layout-color0);
}

.jp-Notebook.jp-mod-scrollPastEnd::after {
  display: block;
  content: '';
  min-height: var(--jp-notebook-scroll-padding);
}

.jp-MainAreaWidget-ContainStrict .jp-Notebook * {
  contain: strict;
}

.jp-Notebook-render * {
  contain: none !important;
}

.jp-Notebook .jp-Cell {
  overflow: visible;
}

.jp-Notebook .jp-Cell .jp-InputPrompt {
  cursor: move;
  float: left;
}

/*-----------------------------------------------------------------------------
| Notebook state related styling
|
| The notebook and cells each have states, here are the possibilities:
|
| - Notebook
|   - Command
|   - Edit
| - Cell
|   - None
|   - Active (only one can be active)
|   - Selected (the cells actions are applied to)
|   - Multiselected (when multiple selected, the cursor)
|   - No outputs
|----------------------------------------------------------------------------*/

/* Command or edit modes */

.jp-Notebook .jp-Cell:not(.jp-mod-active) .jp-InputPrompt {
  opacity: var(--jp-cell-prompt-not-active-opacity);
  color: var(--jp-cell-prompt-not-active-font-color);
}

.jp-Notebook .jp-Cell:not(.jp-mod-active) .jp-OutputPrompt {
  opacity: var(--jp-cell-prompt-not-active-opacity);
  color: var(--jp-cell-prompt-not-active-font-color);
}

/* cell is active */
.jp-Notebook .jp-Cell.jp-mod-active .jp-Collapser {
  background: var(--jp-brand-color1);
}

/* cell is dirty */
.jp-Notebook .jp-Cell.jp-mod-dirty .jp-InputPrompt {
  color: var(--jp-warn-color1);
}
.jp-Notebook .jp-Cell.jp-mod-dirty .jp-InputPrompt:before {
  color: var(--jp-warn-color1);
  content: '•';
}

.jp-Notebook .jp-Cell.jp-mod-active.jp-mod-dirty .jp-Collapser {
  background: var(--jp-warn-color1);
}

/* collapser is hovered */
.jp-Notebook .jp-Cell .jp-Collapser:hover {
  box-shadow: var(--jp-elevation-z2);
  background: var(--jp-brand-color1);
  opacity: var(--jp-cell-collapser-not-active-hover-opacity);
}

/* cell is active and collapser is hovered */
.jp-Notebook .jp-Cell.jp-mod-active .jp-Collapser:hover {
  background: var(--jp-brand-color0);
  opacity: 1;
}

/* Command mode */

.jp-Notebook.jp-mod-commandMode .jp-Cell.jp-mod-selected {
  background: var(--jp-notebook-multiselected-color);
}

.jp-Notebook.jp-mod-commandMode
  .jp-Cell.jp-mod-active.jp-mod-selected:not(.jp-mod-multiSelected) {
  background: transparent;
}

/* Edit mode */

.jp-Notebook.jp-mod-editMode .jp-Cell.jp-mod-active .jp-InputArea-editor {
  border: var(--jp-border-width) solid var(--jp-cell-editor-active-border-color);
  box-shadow: var(--jp-input-box-shadow);
  background-color: var(--jp-cell-editor-active-background);
}

/*-----------------------------------------------------------------------------
| Notebook drag and drop
|----------------------------------------------------------------------------*/

.jp-Notebook-cell.jp-mod-dropSource {
  opacity: 0.5;
}

.jp-Notebook-cell.jp-mod-dropTarget,
.jp-Notebook.jp-mod-commandMode
  .jp-Notebook-cell.jp-mod-active.jp-mod-selected.jp-mod-dropTarget {
  border-top-color: var(--jp-private-notebook-selected-color);
  border-top-style: solid;
  border-top-width: 2px;
}

.jp-dragImage {
  display: block;
  flex-direction: row;
  width: var(--jp-private-notebook-dragImage-width);
  height: var(--jp-private-notebook-dragImage-height);
  border: var(--jp-border-width) solid var(--jp-cell-editor-border-color);
  background: var(--jp-cell-editor-background);
  overflow: visible;
}

.jp-dragImage-singlePrompt {
  box-shadow: 2px 2px 4px 0px rgba(0, 0, 0, 0.12);
}

.jp-dragImage .jp-dragImage-content {
  flex: 1 1 auto;
  z-index: 2;
  font-size: var(--jp-code-font-size);
  font-family: var(--jp-code-font-family);
  line-height: var(--jp-code-line-height);
  padding: var(--jp-code-padding);
  border: var(--jp-border-width) solid var(--jp-cell-editor-border-color);
  background: var(--jp-cell-editor-background-color);
  color: var(--jp-content-font-color3);
  text-align: left;
  margin: 4px 4px 4px 0px;
}

.jp-dragImage .jp-dragImage-prompt {
  flex: 0 0 auto;
  min-width: 36px;
  color: var(--jp-cell-inprompt-font-color);
  padding: var(--jp-code-padding);
  padding-left: 12px;
  font-family: var(--jp-cell-prompt-font-family);
  letter-spacing: var(--jp-cell-prompt-letter-spacing);
  line-height: 1.9;
  font-size: var(--jp-code-font-size);
  border: var(--jp-border-width) solid transparent;
}

.jp-dragImage-multipleBack {
  z-index: -1;
  position: absolute;
  height: 32px;
  width: 300px;
  top: 8px;
  left: 8px;
  background: var(--jp-layout-color2);
  border: var(--jp-border-width) solid var(--jp-input-border-color);
  box-shadow: 2px 2px 4px 0px rgba(0, 0, 0, 0.12);
}

/*-----------------------------------------------------------------------------
| Cell toolbar
|----------------------------------------------------------------------------*/

.jp-NotebookTools {
  display: block;
  min-width: var(--jp-sidebar-min-width);
  color: var(--jp-ui-font-color1);
  background: var(--jp-layout-color1);
  /* This is needed so that all font sizing of children done in ems is
    * relative to this base size */
  font-size: var(--jp-ui-font-size1);
  overflow: auto;
}

.jp-NotebookTools-tool {
  padding: 0px 12px 0 12px;
}

.jp-ActiveCellTool {
  padding: 12px;
  background-color: var(--jp-layout-color1);
  border-top: none !important;
}

.jp-ActiveCellTool .jp-InputArea-prompt {
  flex: 0 0 auto;
  padding-left: 0px;
}

.jp-ActiveCellTool .jp-InputArea-editor {
  flex: 1 1 auto;
  background: var(--jp-cell-editor-background);
  border-color: var(--jp-cell-editor-border-color);
}

.jp-ActiveCellTool .jp-InputArea-editor .CodeMirror {
  background: transparent;
}

.jp-MetadataEditorTool {
  flex-direction: column;
  padding: 12px 0px 12px 0px;
}

.jp-RankedPanel > :not(:first-child) {
  margin-top: 12px;
}

.jp-KeySelector select.jp-mod-styled {
  font-size: var(--jp-ui-font-size1);
  color: var(--jp-ui-font-color0);
  border: var(--jp-border-width) solid var(--jp-border-color1);
}

.jp-KeySelector label,
.jp-MetadataEditorTool label {
  line-height: 1.4;
}

.jp-NotebookTools .jp-select-wrapper {
  margin-top: 4px;
  margin-bottom: 0px;
}

.jp-NotebookTools .jp-Collapse {
  margin-top: 16px;
}

/*-----------------------------------------------------------------------------
| Presentation Mode (.jp-mod-presentationMode)
|----------------------------------------------------------------------------*/

.jp-mod-presentationMode .jp-Notebook {
  --jp-content-font-size1: var(--jp-content-presentation-font-size1);
  --jp-code-font-size: var(--jp-code-presentation-font-size);
}

.jp-mod-presentationMode .jp-Notebook .jp-Cell .jp-InputPrompt,
.jp-mod-presentationMode .jp-Notebook .jp-Cell .jp-OutputPrompt {
  flex: 0 0 110px;
}

/*-----------------------------------------------------------------------------
| Placeholder
|----------------------------------------------------------------------------*/

.jp-Cell-Placeholder {
  padding-left: 55px;
}

.jp-Cell-Placeholder-wrapper {
  background: #fff;
  border: 1px solid;
  border-color: #e5e6e9 #dfe0e4 #d0d1d5;
  border-radius: 4px;
  -webkit-border-radius: 4px;
  margin: 10px 15px;
}

.jp-Cell-Placeholder-wrapper-inner {
  padding: 15px;
  position: relative;
}

.jp-Cell-Placeholder-wrapper-body {
  background-repeat: repeat;
  background-size: 50% auto;
}

.jp-Cell-Placeholder-wrapper-body div {
  background: #f6f7f8;
  background-image: -webkit-linear-gradient(
    left,
    #f6f7f8 0%,
    #edeef1 20%,
    #f6f7f8 40%,
    #f6f7f8 100%
  );
  background-repeat: no-repeat;
  background-size: 800px 104px;
  height: 104px;
  position: relative;
}

.jp-Cell-Placeholder-wrapper-body div {
  position: absolute;
  right: 15px;
  left: 15px;
  top: 15px;
}

div.jp-Cell-Placeholder-h1 {
  top: 20px;
  height: 20px;
  left: 15px;
  width: 150px;
}

div.jp-Cell-Placeholder-h2 {
  left: 15px;
  top: 50px;
  height: 10px;
  width: 100px;
}

div.jp-Cell-Placeholder-content-1,
div.jp-Cell-Placeholder-content-2,
div.jp-Cell-Placeholder-content-3 {
  left: 15px;
  right: 15px;
  height: 10px;
}

div.jp-Cell-Placeholder-content-1 {
  top: 100px;
}

div.jp-Cell-Placeholder-content-2 {
  top: 120px;
}

div.jp-Cell-Placeholder-content-3 {
  top: 140px;
}

</style>

    <style type="text/css">
/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*
The following CSS variables define the main, public API for styling JupyterLab.
These variables should be used by all plugins wherever possible. In other
words, plugins should not define custom colors, sizes, etc unless absolutely
necessary. This enables users to change the visual theme of JupyterLab
by changing these variables.

Many variables appear in an ordered sequence (0,1,2,3). These sequences
are designed to work well together, so for example, `--jp-border-color1` should
be used with `--jp-layout-color1`. The numbers have the following meanings:

* 0: super-primary, reserved for special emphasis
* 1: primary, most important under normal situations
* 2: secondary, next most important under normal situations
* 3: tertiary, next most important under normal situations

Throughout JupyterLab, we are mostly following principles from Google's
Material Design when selecting colors. We are not, however, following
all of MD as it is not optimized for dense, information rich UIs.
*/

:root {
  /* Elevation
   *
   * We style box-shadows using Material Design's idea of elevation. These particular numbers are taken from here:
   *
   * https://github.com/material-components/material-components-web
   * https://material-components-web.appspot.com/elevation.html
   */

  --jp-shadow-base-lightness: 0;
  --jp-shadow-umbra-color: rgba(
    var(--jp-shadow-base-lightness),
    var(--jp-shadow-base-lightness),
    var(--jp-shadow-base-lightness),
    0.2
  );
  --jp-shadow-penumbra-color: rgba(
    var(--jp-shadow-base-lightness),
    var(--jp-shadow-base-lightness),
    var(--jp-shadow-base-lightness),
    0.14
  );
  --jp-shadow-ambient-color: rgba(
    var(--jp-shadow-base-lightness),
    var(--jp-shadow-base-lightness),
    var(--jp-shadow-base-lightness),
    0.12
  );
  --jp-elevation-z0: none;
  --jp-elevation-z1: 0px 2px 1px -1px var(--jp-shadow-umbra-color),
    0px 1px 1px 0px var(--jp-shadow-penumbra-color),
    0px 1px 3px 0px var(--jp-shadow-ambient-color);
  --jp-elevation-z2: 0px 3px 1px -2px var(--jp-shadow-umbra-color),
    0px 2px 2px 0px var(--jp-shadow-penumbra-color),
    0px 1px 5px 0px var(--jp-shadow-ambient-color);
  --jp-elevation-z4: 0px 2px 4px -1px var(--jp-shadow-umbra-color),
    0px 4px 5px 0px var(--jp-shadow-penumbra-color),
    0px 1px 10px 0px var(--jp-shadow-ambient-color);
  --jp-elevation-z6: 0px 3px 5px -1px var(--jp-shadow-umbra-color),
    0px 6px 10px 0px var(--jp-shadow-penumbra-color),
    0px 1px 18px 0px var(--jp-shadow-ambient-color);
  --jp-elevation-z8: 0px 5px 5px -3px var(--jp-shadow-umbra-color),
    0px 8px 10px 1px var(--jp-shadow-penumbra-color),
    0px 3px 14px 2px var(--jp-shadow-ambient-color);
  --jp-elevation-z12: 0px 7px 8px -4px var(--jp-shadow-umbra-color),
    0px 12px 17px 2px var(--jp-shadow-penumbra-color),
    0px 5px 22px 4px var(--jp-shadow-ambient-color);
  --jp-elevation-z16: 0px 8px 10px -5px var(--jp-shadow-umbra-color),
    0px 16px 24px 2px var(--jp-shadow-penumbra-color),
    0px 6px 30px 5px var(--jp-shadow-ambient-color);
  --jp-elevation-z20: 0px 10px 13px -6px var(--jp-shadow-umbra-color),
    0px 20px 31px 3px var(--jp-shadow-penumbra-color),
    0px 8px 38px 7px var(--jp-shadow-ambient-color);
  --jp-elevation-z24: 0px 11px 15px -7px var(--jp-shadow-umbra-color),
    0px 24px 38px 3px var(--jp-shadow-penumbra-color),
    0px 9px 46px 8px var(--jp-shadow-ambient-color);

  /* Borders
   *
   * The following variables, specify the visual styling of borders in JupyterLab.
   */

  --jp-border-width: 1px;
  --jp-border-color0: var(--md-grey-400);
  --jp-border-color1: var(--md-grey-400);
  --jp-border-color2: var(--md-grey-300);
  --jp-border-color3: var(--md-grey-200);
  --jp-border-radius: 2px;

  /* UI Fonts
   *
   * The UI font CSS variables are used for the typography all of the JupyterLab
   * user interface elements that are not directly user generated content.
   *
   * The font sizing here is done assuming that the body font size of --jp-ui-font-size1
   * is applied to a parent element. When children elements, such as headings, are sized
   * in em all things will be computed relative to that body size.
   */

  --jp-ui-font-scale-factor: 1.2;
  --jp-ui-font-size0: 0.83333em;
  --jp-ui-font-size1: 13px; /* Base font size */
  --jp-ui-font-size2: 1.2em;
  --jp-ui-font-size3: 1.44em;

  --jp-ui-font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica,
    Arial, sans-serif, 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol';

  /*
   * Use these font colors against the corresponding main layout colors.
   * In a light theme, these go from dark to light.
   */

  /* Defaults use Material Design specification */
  --jp-ui-font-color0: rgba(0, 0, 0, 1);
  --jp-ui-font-color1: rgba(0, 0, 0, 0.87);
  --jp-ui-font-color2: rgba(0, 0, 0, 0.54);
  --jp-ui-font-color3: rgba(0, 0, 0, 0.38);

  /*
   * Use these against the brand/accent/warn/error colors.
   * These will typically go from light to darker, in both a dark and light theme.
   */

  --jp-ui-inverse-font-color0: rgba(255, 255, 255, 1);
  --jp-ui-inverse-font-color1: rgba(255, 255, 255, 1);
  --jp-ui-inverse-font-color2: rgba(255, 255, 255, 0.7);
  --jp-ui-inverse-font-color3: rgba(255, 255, 255, 0.5);

  /* Content Fonts
   *
   * Content font variables are used for typography of user generated content.
   *
   * The font sizing here is done assuming that the body font size of --jp-content-font-size1
   * is applied to a parent element. When children elements, such as headings, are sized
   * in em all things will be computed relative to that body size.
   */

  --jp-content-line-height: 1.6;
  --jp-content-font-scale-factor: 1.2;
  --jp-content-font-size0: 0.83333em;
  --jp-content-font-size1: 14px; /* Base font size */
  --jp-content-font-size2: 1.2em;
  --jp-content-font-size3: 1.44em;
  --jp-content-font-size4: 1.728em;
  --jp-content-font-size5: 2.0736em;

  /* This gives a magnification of about 125% in presentation mode over normal. */
  --jp-content-presentation-font-size1: 17px;

  --jp-content-heading-line-height: 1;
  --jp-content-heading-margin-top: 1.2em;
  --jp-content-heading-margin-bottom: 0.8em;
  --jp-content-heading-font-weight: 500;

  /* Defaults use Material Design specification */
  --jp-content-font-color0: rgba(0, 0, 0, 1);
  --jp-content-font-color1: rgba(0, 0, 0, 0.87);
  --jp-content-font-color2: rgba(0, 0, 0, 0.54);
  --jp-content-font-color3: rgba(0, 0, 0, 0.38);

  --jp-content-link-color: var(--md-blue-700);

  --jp-content-font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI',
    Helvetica, Arial, sans-serif, 'Apple Color Emoji', 'Segoe UI Emoji',
    'Segoe UI Symbol';

  /*
   * Code Fonts
   *
   * Code font variables are used for typography of code and other monospaces content.
   */

  --jp-code-font-size: 13px;
  --jp-code-line-height: 1.3077; /* 17px for 13px base */
  --jp-code-padding: 5px; /* 5px for 13px base, codemirror highlighting needs integer px value */
  --jp-code-font-family-default: Menlo, Consolas, 'DejaVu Sans Mono', monospace;
  --jp-code-font-family: var(--jp-code-font-family-default);

  /* This gives a magnification of about 125% in presentation mode over normal. */
  --jp-code-presentation-font-size: 16px;

  /* may need to tweak cursor width if you change font size */
  --jp-code-cursor-width0: 1.4px;
  --jp-code-cursor-width1: 2px;
  --jp-code-cursor-width2: 4px;

  /* Layout
   *
   * The following are the main layout colors use in JupyterLab. In a light
   * theme these would go from light to dark.
   */

  --jp-layout-color0: white;
  --jp-layout-color1: white;
  --jp-layout-color2: var(--md-grey-200);
  --jp-layout-color3: var(--md-grey-400);
  --jp-layout-color4: var(--md-grey-600);

  /* Inverse Layout
   *
   * The following are the inverse layout colors use in JupyterLab. In a light
   * theme these would go from dark to light.
   */

  --jp-inverse-layout-color0: #111111;
  --jp-inverse-layout-color1: var(--md-grey-900);
  --jp-inverse-layout-color2: var(--md-grey-800);
  --jp-inverse-layout-color3: var(--md-grey-700);
  --jp-inverse-layout-color4: var(--md-grey-600);

  /* Brand/accent */

  --jp-brand-color0: var(--md-blue-900);
  --jp-brand-color1: var(--md-blue-700);
  --jp-brand-color2: var(--md-blue-300);
  --jp-brand-color3: var(--md-blue-100);
  --jp-brand-color4: var(--md-blue-50);

  --jp-accent-color0: var(--md-green-900);
  --jp-accent-color1: var(--md-green-700);
  --jp-accent-color2: var(--md-green-300);
  --jp-accent-color3: var(--md-green-100);

  /* State colors (warn, error, success, info) */

  --jp-warn-color0: var(--md-orange-900);
  --jp-warn-color1: var(--md-orange-700);
  --jp-warn-color2: var(--md-orange-300);
  --jp-warn-color3: var(--md-orange-100);

  --jp-error-color0: var(--md-red-900);
  --jp-error-color1: var(--md-red-700);
  --jp-error-color2: var(--md-red-300);
  --jp-error-color3: var(--md-red-100);

  --jp-success-color0: var(--md-green-900);
  --jp-success-color1: var(--md-green-700);
  --jp-success-color2: var(--md-green-300);
  --jp-success-color3: var(--md-green-100);

  --jp-info-color0: var(--md-cyan-900);
  --jp-info-color1: var(--md-cyan-700);
  --jp-info-color2: var(--md-cyan-300);
  --jp-info-color3: var(--md-cyan-100);

  /* Cell specific styles */

  --jp-cell-padding: 5px;

  --jp-cell-collapser-width: 8px;
  --jp-cell-collapser-min-height: 20px;
  --jp-cell-collapser-not-active-hover-opacity: 0.6;

  --jp-cell-editor-background: var(--md-grey-100);
  --jp-cell-editor-border-color: var(--md-grey-300);
  --jp-cell-editor-box-shadow: inset 0 0 2px var(--md-blue-300);
  --jp-cell-editor-active-background: var(--jp-layout-color0);
  --jp-cell-editor-active-border-color: var(--jp-brand-color1);

  --jp-cell-prompt-width: 64px;
  --jp-cell-prompt-font-family: var(--jp-code-font-family-default);
  --jp-cell-prompt-letter-spacing: 0px;
  --jp-cell-prompt-opacity: 1;
  --jp-cell-prompt-not-active-opacity: 0.5;
  --jp-cell-prompt-not-active-font-color: var(--md-grey-700);
  /* A custom blend of MD grey and blue 600
   * See https://meyerweb.com/eric/tools/color-blend/#546E7A:1E88E5:5:hex */
  --jp-cell-inprompt-font-color: #307fc1;
  /* A custom blend of MD grey and orange 600
   * https://meyerweb.com/eric/tools/color-blend/#546E7A:F4511E:5:hex */
  --jp-cell-outprompt-font-color: #bf5b3d;

  /* Notebook specific styles */

  --jp-notebook-padding: 10px;
  --jp-notebook-select-background: var(--jp-layout-color1);
  --jp-notebook-multiselected-color: var(--md-blue-50);

  /* The scroll padding is calculated to fill enough space at the bottom of the
  notebook to show one single-line cell (with appropriate padding) at the top
  when the notebook is scrolled all the way to the bottom. We also subtract one
  pixel so that no scrollbar appears if we have just one single-line cell in the
  notebook. This padding is to enable a 'scroll past end' feature in a notebook.
  */
  --jp-notebook-scroll-padding: calc(
    100% - var(--jp-code-font-size) * var(--jp-code-line-height) -
      var(--jp-code-padding) - var(--jp-cell-padding) - 1px
  );

  /* Rendermime styles */

  --jp-rendermime-error-background: #fdd;
  --jp-rendermime-table-row-background: var(--md-grey-100);
  --jp-rendermime-table-row-hover-background: var(--md-light-blue-50);

  /* Dialog specific styles */

  --jp-dialog-background: rgba(0, 0, 0, 0.25);

  /* Console specific styles */

  --jp-console-padding: 10px;

  /* Toolbar specific styles */

  --jp-toolbar-border-color: var(--jp-border-color1);
  --jp-toolbar-micro-height: 8px;
  --jp-toolbar-background: var(--jp-layout-color1);
  --jp-toolbar-box-shadow: 0px 0px 2px 0px rgba(0, 0, 0, 0.24);
  --jp-toolbar-header-margin: 4px 4px 0px 4px;
  --jp-toolbar-active-background: var(--md-grey-300);

  /* Statusbar specific styles */

  --jp-statusbar-height: 24px;

  /* Input field styles */

  --jp-input-box-shadow: inset 0 0 2px var(--md-blue-300);
  --jp-input-active-background: var(--jp-layout-color1);
  --jp-input-hover-background: var(--jp-layout-color1);
  --jp-input-background: var(--md-grey-100);
  --jp-input-border-color: var(--jp-border-color1);
  --jp-input-active-border-color: var(--jp-brand-color1);
  --jp-input-active-box-shadow-color: rgba(19, 124, 189, 0.3);

  /* General editor styles */

  --jp-editor-selected-background: #d9d9d9;
  --jp-editor-selected-focused-background: #d7d4f0;
  --jp-editor-cursor-color: var(--jp-ui-font-color0);

  /* Code mirror specific styles */

  --jp-mirror-editor-keyword-color: #008000;
  --jp-mirror-editor-atom-color: #88f;
  --jp-mirror-editor-number-color: #080;
  --jp-mirror-editor-def-color: #00f;
  --jp-mirror-editor-variable-color: var(--md-grey-900);
  --jp-mirror-editor-variable-2-color: #05a;
  --jp-mirror-editor-variable-3-color: #085;
  --jp-mirror-editor-punctuation-color: #05a;
  --jp-mirror-editor-property-color: #05a;
  --jp-mirror-editor-operator-color: #aa22ff;
  --jp-mirror-editor-comment-color: #408080;
  --jp-mirror-editor-string-color: #ba2121;
  --jp-mirror-editor-string-2-color: #708;
  --jp-mirror-editor-meta-color: #aa22ff;
  --jp-mirror-editor-qualifier-color: #555;
  --jp-mirror-editor-builtin-color: #008000;
  --jp-mirror-editor-bracket-color: #997;
  --jp-mirror-editor-tag-color: #170;
  --jp-mirror-editor-attribute-color: #00c;
  --jp-mirror-editor-header-color: blue;
  --jp-mirror-editor-quote-color: #090;
  --jp-mirror-editor-link-color: #00c;
  --jp-mirror-editor-error-color: #f00;
  --jp-mirror-editor-hr-color: #999;

  /* Vega extension styles */

  --jp-vega-background: white;

  /* Sidebar-related styles */

  --jp-sidebar-min-width: 250px;

  /* Search-related styles */

  --jp-search-toggle-off-opacity: 0.5;
  --jp-search-toggle-hover-opacity: 0.8;
  --jp-search-toggle-on-opacity: 1;
  --jp-search-selected-match-background-color: rgb(245, 200, 0);
  --jp-search-selected-match-color: black;
  --jp-search-unselected-match-background-color: var(
    --jp-inverse-layout-color0
  );
  --jp-search-unselected-match-color: var(--jp-ui-inverse-font-color0);

  /* Icon colors that work well with light or dark backgrounds */
  --jp-icon-contrast-color0: var(--md-purple-600);
  --jp-icon-contrast-color1: var(--md-green-600);
  --jp-icon-contrast-color2: var(--md-pink-600);
  --jp-icon-contrast-color3: var(--md-blue-600);
}
</style>

<style type="text/css">
/* Force rendering true colors when outputing to pdf */
* {
  -webkit-print-color-adjust: exact;
}

/* Misc */
a.anchor-link {
  display: none;
}

.highlight  {
  margin: 0.4em;
}

/* Input area styling */
.jp-InputArea {
  overflow: hidden;
}

.jp-InputArea-editor {
  overflow: hidden;
}

.CodeMirror pre {
  margin: 0;
  padding: 0;
}

/* Using table instead of flexbox so that we can use break-inside property */
/* CSS rules under this comment should not be required anymore after we move to the JupyterLab 4.0 CSS */


.jp-CodeCell.jp-mod-outputsScrolled .jp-OutputArea-prompt {
  min-width: calc(
    var(--jp-cell-prompt-width) - var(--jp-private-cell-scrolling-output-offset)
  );
}

.jp-OutputArea-child {
  display: table;
  width: 100%;
}

.jp-OutputPrompt {
  display: table-cell;
  vertical-align: top;
  min-width: var(--jp-cell-prompt-width);
}

body[data-format='mobile'] .jp-OutputPrompt {
  display: table-row;
}

.jp-OutputArea-output {
  display: table-cell;
  width: 100%;
}

body[data-format='mobile'] .jp-OutputArea-child .jp-OutputArea-output {
  display: table-row;
}

.jp-OutputArea-output.jp-OutputArea-executeResult {
  width: 100%;
}

/* Hiding the collapser by default */
.jp-Collapser {
  display: none;
}

@media print {
  .jp-Cell-inputWrapper,
  .jp-Cell-outputWrapper {
    display: block;
  }

  .jp-OutputArea-child {
    break-inside: avoid-page;
  }
}
</style>

<!-- Load mathjax -->
    <script type="text/javascript" async="" src="./Note_files/MathJax.js.download"></script>
    <!-- MathJax configuration -->
    <script type="text/x-mathjax-config;executed=true">
    init_mathjax = function() {
        if (window.MathJax) {
        // MathJax loaded
            MathJax.Hub.Config({
                TeX: {
                    equationNumbers: {
                    autoNumber: "AMS",
                    useLabelIds: true
                    }
                },
                tex2jax: {
                    inlineMath: [ ['$','$'], ["\\(","\\)"] ],
                    displayMath: [ ['$$','$$'], ["\\[","\\]"] ],
                    processEscapes: true,
                    processEnvironments: true
                },
                displayAlign: 'center',
                CommonHTML: {
                    linebreaks: {
                    automatic: true
                    }
                }
            });

            MathJax.Hub.Queue(["Typeset", MathJax.Hub]);
        }
    }
    init_mathjax();
    </script>
    <!-- End of mathjax configuration --><style type="text/css">.MathJax_Hover_Frame {border-radius: .25em; -webkit-border-radius: .25em; -moz-border-radius: .25em; -khtml-border-radius: .25em; box-shadow: 0px 0px 15px #83A; -webkit-box-shadow: 0px 0px 15px #83A; -moz-box-shadow: 0px 0px 15px #83A; -khtml-box-shadow: 0px 0px 15px #83A; border: 1px solid #A6D ! important; display: inline-block; position: absolute}
.MathJax_Menu_Button .MathJax_Hover_Arrow {position: absolute; cursor: pointer; display: inline-block; border: 2px solid #AAA; border-radius: 4px; -webkit-border-radius: 4px; -moz-border-radius: 4px; -khtml-border-radius: 4px; font-family: 'Courier New',Courier; font-size: 9px; color: #F0F0F0}
.MathJax_Menu_Button .MathJax_Hover_Arrow span {display: block; background-color: #AAA; border: 1px solid; border-radius: 3px; line-height: 0; padding: 4px}
.MathJax_Hover_Arrow:hover {color: white!important; border: 2px solid #CCC!important}
.MathJax_Hover_Arrow:hover span {background-color: #CCC!important}
</style><style type="text/css">#MathJax_About {position: fixed; left: 50%; width: auto; text-align: center; border: 3px outset; padding: 1em 2em; background-color: #DDDDDD; color: black; cursor: default; font-family: message-box; font-size: 120%; font-style: normal; text-indent: 0; text-transform: none; line-height: normal; letter-spacing: normal; word-spacing: normal; word-wrap: normal; white-space: nowrap; float: none; z-index: 201; border-radius: 15px; -webkit-border-radius: 15px; -moz-border-radius: 15px; -khtml-border-radius: 15px; box-shadow: 0px 10px 20px #808080; -webkit-box-shadow: 0px 10px 20px #808080; -moz-box-shadow: 0px 10px 20px #808080; -khtml-box-shadow: 0px 10px 20px #808080; filter: progid:DXImageTransform.Microsoft.dropshadow(OffX=2, OffY=2, Color='gray', Positive='true')}
#MathJax_About.MathJax_MousePost {outline: none}
.MathJax_Menu {position: absolute; background-color: white; color: black; width: auto; padding: 2px; border: 1px solid #CCCCCC; margin: 0; cursor: default; font: menu; text-align: left; text-indent: 0; text-transform: none; line-height: normal; letter-spacing: normal; word-spacing: normal; word-wrap: normal; white-space: nowrap; float: none; z-index: 201; box-shadow: 0px 10px 20px #808080; -webkit-box-shadow: 0px 10px 20px #808080; -moz-box-shadow: 0px 10px 20px #808080; -khtml-box-shadow: 0px 10px 20px #808080; filter: progid:DXImageTransform.Microsoft.dropshadow(OffX=2, OffY=2, Color='gray', Positive='true')}
.MathJax_MenuItem {padding: 2px 2em; background: transparent}
.MathJax_MenuArrow {position: absolute; right: .5em; padding-top: .25em; color: #666666; font-size: .75em}
.MathJax_MenuActive .MathJax_MenuArrow {color: white}
.MathJax_MenuArrow.RTL {left: .5em; right: auto}
.MathJax_MenuCheck {position: absolute; left: .7em}
.MathJax_MenuCheck.RTL {right: .7em; left: auto}
.MathJax_MenuRadioCheck {position: absolute; left: 1em}
.MathJax_MenuRadioCheck.RTL {right: 1em; left: auto}
.MathJax_MenuLabel {padding: 2px 2em 4px 1.33em; font-style: italic}
.MathJax_MenuRule {border-top: 1px solid #CCCCCC; margin: 4px 1px 0px}
.MathJax_MenuDisabled {color: GrayText}
.MathJax_MenuActive {background-color: Highlight; color: HighlightText}
.MathJax_MenuDisabled:focus, .MathJax_MenuLabel:focus {background-color: #E8E8E8}
.MathJax_ContextMenu:focus {outline: none}
.MathJax_ContextMenu .MathJax_MenuItem:focus {outline: none}
#MathJax_AboutClose {top: .2em; right: .2em}
.MathJax_Menu .MathJax_MenuClose {top: -10px; left: -10px}
.MathJax_MenuClose {position: absolute; cursor: pointer; display: inline-block; border: 2px solid #AAA; border-radius: 18px; -webkit-border-radius: 18px; -moz-border-radius: 18px; -khtml-border-radius: 18px; font-family: 'Courier New',Courier; font-size: 24px; color: #F0F0F0}
.MathJax_MenuClose span {display: block; background-color: #AAA; border: 1.5px solid; border-radius: 18px; -webkit-border-radius: 18px; -moz-border-radius: 18px; -khtml-border-radius: 18px; line-height: 0; padding: 8px 0 6px}
.MathJax_MenuClose:hover {color: white!important; border: 2px solid #CCC!important}
.MathJax_MenuClose:hover span {background-color: #CCC!important}
.MathJax_MenuClose:hover:focus {outline: none}
</style><style type="text/css">.MathJax_Preview .MJXf-math {color: inherit!important}
</style><style type="text/css">.MJX_Assistive_MathML {position: absolute!important; top: 0; left: 0; clip: rect(1px, 1px, 1px, 1px); padding: 1px 0 0 0!important; border: 0!important; height: 1px!important; width: 1px!important; overflow: hidden!important; display: block!important; -webkit-touch-callout: none; -webkit-user-select: none; -khtml-user-select: none; -moz-user-select: none; -ms-user-select: none; user-select: none}
.MJX_Assistive_MathML.MJX_Assistive_MathML_Block {width: 100%!important}
</style><style type="text/css">#MathJax_Zoom {position: absolute; background-color: #F0F0F0; overflow: auto; display: block; z-index: 301; padding: .5em; border: 1px solid black; margin: 0; font-weight: normal; font-style: normal; text-align: left; text-indent: 0; text-transform: none; line-height: normal; letter-spacing: normal; word-spacing: normal; word-wrap: normal; white-space: nowrap; float: none; -webkit-box-sizing: content-box; -moz-box-sizing: content-box; box-sizing: content-box; box-shadow: 5px 5px 15px #AAAAAA; -webkit-box-shadow: 5px 5px 15px #AAAAAA; -moz-box-shadow: 5px 5px 15px #AAAAAA; -khtml-box-shadow: 5px 5px 15px #AAAAAA; filter: progid:DXImageTransform.Microsoft.dropshadow(OffX=2, OffY=2, Color='gray', Positive='true')}
#MathJax_ZoomOverlay {position: absolute; left: 0; top: 0; z-index: 300; display: inline-block; width: 100%; height: 100%; border: 0; padding: 0; margin: 0; background-color: white; opacity: 0; filter: alpha(opacity=0)}
#MathJax_ZoomFrame {position: relative; display: inline-block; height: 0; width: 0}
#MathJax_ZoomEventTrap {position: absolute; left: 0; top: 0; z-index: 302; display: inline-block; border: 0; padding: 0; margin: 0; background-color: white; opacity: 0; filter: alpha(opacity=0)}
</style><style type="text/css">.MathJax_Preview {color: #888}
#MathJax_Message {position: fixed; left: 1em; bottom: 1.5em; background-color: #E6E6E6; border: 1px solid #959595; margin: 0px; padding: 2px 8px; z-index: 102; color: black; font-size: 80%; width: auto; white-space: nowrap}
#MathJax_MSIE_Frame {position: absolute; top: 0; left: 0; width: 0px; z-index: 101; border: 0px; margin: 0px; padding: 0px}
.MathJax_Error {color: #CC0000; font-style: italic}
</style><style type="text/css">.MJXp-script {font-size: .8em}
.MJXp-right {-webkit-transform-origin: right; -moz-transform-origin: right; -ms-transform-origin: right; -o-transform-origin: right; transform-origin: right}
.MJXp-bold {font-weight: bold}
.MJXp-italic {font-style: italic}
.MJXp-scr {font-family: MathJax_Script,'Times New Roman',Times,STIXGeneral,serif}
.MJXp-frak {font-family: MathJax_Fraktur,'Times New Roman',Times,STIXGeneral,serif}
.MJXp-sf {font-family: MathJax_SansSerif,'Times New Roman',Times,STIXGeneral,serif}
.MJXp-cal {font-family: MathJax_Caligraphic,'Times New Roman',Times,STIXGeneral,serif}
.MJXp-mono {font-family: MathJax_Typewriter,'Times New Roman',Times,STIXGeneral,serif}
.MJXp-largeop {font-size: 150%}
.MJXp-largeop.MJXp-int {vertical-align: -.2em}
.MJXp-math {display: inline-block; line-height: 1.2; text-indent: 0; font-family: 'Times New Roman',Times,STIXGeneral,serif; white-space: nowrap; border-collapse: collapse}
.MJXp-display {display: block; text-align: center; margin: 1em 0}
.MJXp-math span {display: inline-block}
.MJXp-box {display: block!important; text-align: center}
.MJXp-box:after {content: " "}
.MJXp-rule {display: block!important; margin-top: .1em}
.MJXp-char {display: block!important}
.MJXp-mo {margin: 0 .15em}
.MJXp-mfrac {margin: 0 .125em; vertical-align: .25em}
.MJXp-denom {display: inline-table!important; width: 100%}
.MJXp-denom > * {display: table-row!important}
.MJXp-surd {vertical-align: top}
.MJXp-surd > * {display: block!important}
.MJXp-script-box > *  {display: table!important; height: 50%}
.MJXp-script-box > * > * {display: table-cell!important; vertical-align: top}
.MJXp-script-box > *:last-child > * {vertical-align: bottom}
.MJXp-script-box > * > * > * {display: block!important}
.MJXp-mphantom {visibility: hidden}
.MJXp-munderover, .MJXp-munder {display: inline-table!important}
.MJXp-over {display: inline-block!important; text-align: center}
.MJXp-over > * {display: block!important}
.MJXp-munderover > *, .MJXp-munder > * {display: table-row!important}
.MJXp-mtable {vertical-align: .25em; margin: 0 .125em}
.MJXp-mtable > * {display: inline-table!important; vertical-align: middle}
.MJXp-mtr {display: table-row!important}
.MJXp-mtd {display: table-cell!important; text-align: center; padding: .5em 0 0 .5em}
.MJXp-mtr > .MJXp-mtd:first-child {padding-left: 0}
.MJXp-mtr:first-child > .MJXp-mtd {padding-top: 0}
.MJXp-mlabeledtr {display: table-row!important}
.MJXp-mlabeledtr > .MJXp-mtd:first-child {padding-left: 0}
.MJXp-mlabeledtr:first-child > .MJXp-mtd {padding-top: 0}
.MJXp-merror {background-color: #FFFF88; color: #CC0000; border: 1px solid #CC0000; padding: 1px 3px; font-style: normal; font-size: 90%}
.MJXp-scale0 {-webkit-transform: scaleX(.0); -moz-transform: scaleX(.0); -ms-transform: scaleX(.0); -o-transform: scaleX(.0); transform: scaleX(.0)}
.MJXp-scale1 {-webkit-transform: scaleX(.1); -moz-transform: scaleX(.1); -ms-transform: scaleX(.1); -o-transform: scaleX(.1); transform: scaleX(.1)}
.MJXp-scale2 {-webkit-transform: scaleX(.2); -moz-transform: scaleX(.2); -ms-transform: scaleX(.2); -o-transform: scaleX(.2); transform: scaleX(.2)}
.MJXp-scale3 {-webkit-transform: scaleX(.3); -moz-transform: scaleX(.3); -ms-transform: scaleX(.3); -o-transform: scaleX(.3); transform: scaleX(.3)}
.MJXp-scale4 {-webkit-transform: scaleX(.4); -moz-transform: scaleX(.4); -ms-transform: scaleX(.4); -o-transform: scaleX(.4); transform: scaleX(.4)}
.MJXp-scale5 {-webkit-transform: scaleX(.5); -moz-transform: scaleX(.5); -ms-transform: scaleX(.5); -o-transform: scaleX(.5); transform: scaleX(.5)}
.MJXp-scale6 {-webkit-transform: scaleX(.6); -moz-transform: scaleX(.6); -ms-transform: scaleX(.6); -o-transform: scaleX(.6); transform: scaleX(.6)}
.MJXp-scale7 {-webkit-transform: scaleX(.7); -moz-transform: scaleX(.7); -ms-transform: scaleX(.7); -o-transform: scaleX(.7); transform: scaleX(.7)}
.MJXp-scale8 {-webkit-transform: scaleX(.8); -moz-transform: scaleX(.8); -ms-transform: scaleX(.8); -o-transform: scaleX(.8); transform: scaleX(.8)}
.MJXp-scale9 {-webkit-transform: scaleX(.9); -moz-transform: scaleX(.9); -ms-transform: scaleX(.9); -o-transform: scaleX(.9); transform: scaleX(.9)}
.MathJax_PHTML .noError {vertical-align: ; font-size: 90%; text-align: left; color: black; padding: 1px 3px; border: 1px solid}
</style><style type="text/css">.mjx-chtml {display: inline-block; line-height: 0; text-indent: 0; text-align: left; text-transform: none; font-style: normal; font-weight: normal; font-size: 100%; font-size-adjust: none; letter-spacing: normal; word-wrap: normal; word-spacing: normal; white-space: nowrap; float: none; direction: ltr; max-width: none; max-height: none; min-width: 0; min-height: 0; border: 0; margin: 0; padding: 1px 0}
.MJXc-display {display: block; text-align: center; margin: 1em 0; padding: 0}
.mjx-chtml[tabindex]:focus, body :focus .mjx-chtml[tabindex] {display: inline-table}
.mjx-full-width {text-align: center; display: table-cell!important; width: 10000em}
.mjx-math {display: inline-block; border-collapse: separate; border-spacing: 0}
.mjx-math * {display: inline-block; -webkit-box-sizing: content-box!important; -moz-box-sizing: content-box!important; box-sizing: content-box!important; text-align: left}
.mjx-numerator {display: block; text-align: center}
.mjx-denominator {display: block; text-align: center}
.MJXc-stacked {height: 0; position: relative}
.MJXc-stacked > * {position: absolute}
.MJXc-bevelled > * {display: inline-block}
.mjx-stack {display: inline-block}
.mjx-op {display: block}
.mjx-under {display: table-cell}
.mjx-over {display: block}
.mjx-over > * {padding-left: 0px!important; padding-right: 0px!important}
.mjx-under > * {padding-left: 0px!important; padding-right: 0px!important}
.mjx-stack > .mjx-sup {display: block}
.mjx-stack > .mjx-sub {display: block}
.mjx-prestack > .mjx-presup {display: block}
.mjx-prestack > .mjx-presub {display: block}
.mjx-delim-h > .mjx-char {display: inline-block}
.mjx-surd {vertical-align: top}
.mjx-mphantom * {visibility: hidden}
.mjx-merror {background-color: #FFFF88; color: #CC0000; border: 1px solid #CC0000; padding: 2px 3px; font-style: normal; font-size: 90%}
.mjx-annotation-xml {line-height: normal}
.mjx-menclose > svg {fill: none; stroke: currentColor}
.mjx-mtr {display: table-row}
.mjx-mlabeledtr {display: table-row}
.mjx-mtd {display: table-cell; text-align: center}
.mjx-label {display: table-row}
.mjx-box {display: inline-block}
.mjx-block {display: block}
.mjx-span {display: inline}
.mjx-char {display: block; white-space: pre}
.mjx-itable {display: inline-table; width: auto}
.mjx-row {display: table-row}
.mjx-cell {display: table-cell}
.mjx-table {display: table; width: 100%}
.mjx-line {display: block; height: 0}
.mjx-strut {width: 0; padding-top: 1em}
.mjx-vsize {width: 0}
.MJXc-space1 {margin-left: .167em}
.MJXc-space2 {margin-left: .222em}
.MJXc-space3 {margin-left: .278em}
.mjx-chartest {display: block; visibility: hidden; position: absolute; top: 0; line-height: normal; font-size: 500%}
.mjx-chartest .mjx-char {display: inline}
.mjx-chartest .mjx-box {padding-top: 1000px}
.MJXc-processing {visibility: hidden; position: fixed; width: 0; height: 0; overflow: hidden}
.MJXc-processed {display: none}
.mjx-test {font-style: normal; font-weight: normal; font-size: 100%; font-size-adjust: none; text-indent: 0; text-transform: none; letter-spacing: normal; word-spacing: normal; overflow: hidden; height: 1px}
.mjx-test.mjx-test-display {display: table!important}
.mjx-test.mjx-test-inline {display: inline!important; margin-right: -1px}
.mjx-test.mjx-test-default {display: block!important; clear: both}
.mjx-ex-box {display: inline-block!important; position: absolute; overflow: hidden; min-height: 0; max-height: none; padding: 0; border: 0; margin: 0; width: 1px; height: 60ex}
.mjx-test-inline .mjx-left-box {display: inline-block; width: 0; float: left}
.mjx-test-inline .mjx-right-box {display: inline-block; width: 0; float: right}
.mjx-test-display .mjx-right-box {display: table-cell!important; width: 10000em!important; min-width: 0; max-width: none; padding: 0; border: 0; margin: 0}
#MathJax_CHTML_Tooltip {background-color: InfoBackground; color: InfoText; border: 1px solid black; box-shadow: 2px 2px 5px #AAAAAA; -webkit-box-shadow: 2px 2px 5px #AAAAAA; -moz-box-shadow: 2px 2px 5px #AAAAAA; -khtml-box-shadow: 2px 2px 5px #AAAAAA; padding: 3px 4px; z-index: 401; position: absolute; left: 0; top: 0; width: auto; height: auto; display: none}
.mjx-chtml .mjx-noError {line-height: 1.2; vertical-align: ; font-size: 90%; text-align: left; color: black; padding: 1px 3px; border: 1px solid}
.MJXc-TeX-unknown-R {font-family: STIXGeneral,'Cambria Math','Arial Unicode MS',serif; font-style: normal; font-weight: normal}
.MJXc-TeX-unknown-I {font-family: STIXGeneral,'Cambria Math','Arial Unicode MS',serif; font-style: italic; font-weight: normal}
.MJXc-TeX-unknown-B {font-family: STIXGeneral,'Cambria Math','Arial Unicode MS',serif; font-style: normal; font-weight: bold}
.MJXc-TeX-unknown-BI {font-family: STIXGeneral,'Cambria Math','Arial Unicode MS',serif; font-style: italic; font-weight: bold}
.MJXc-TeX-ams-R {font-family: MJXc-TeX-ams-R,MJXc-TeX-ams-Rw}
.MJXc-TeX-cal-B {font-family: MJXc-TeX-cal-B,MJXc-TeX-cal-Bx,MJXc-TeX-cal-Bw}
.MJXc-TeX-frak-R {font-family: MJXc-TeX-frak-R,MJXc-TeX-frak-Rw}
.MJXc-TeX-frak-B {font-family: MJXc-TeX-frak-B,MJXc-TeX-frak-Bx,MJXc-TeX-frak-Bw}
.MJXc-TeX-math-BI {font-family: MJXc-TeX-math-BI,MJXc-TeX-math-BIx,MJXc-TeX-math-BIw}
.MJXc-TeX-sans-R {font-family: MJXc-TeX-sans-R,MJXc-TeX-sans-Rw}
.MJXc-TeX-sans-B {font-family: MJXc-TeX-sans-B,MJXc-TeX-sans-Bx,MJXc-TeX-sans-Bw}
.MJXc-TeX-sans-I {font-family: MJXc-TeX-sans-I,MJXc-TeX-sans-Ix,MJXc-TeX-sans-Iw}
.MJXc-TeX-script-R {font-family: MJXc-TeX-script-R,MJXc-TeX-script-Rw}
.MJXc-TeX-type-R {font-family: MJXc-TeX-type-R,MJXc-TeX-type-Rw}
.MJXc-TeX-cal-R {font-family: MJXc-TeX-cal-R,MJXc-TeX-cal-Rw}
.MJXc-TeX-main-B {font-family: MJXc-TeX-main-B,MJXc-TeX-main-Bx,MJXc-TeX-main-Bw}
.MJXc-TeX-main-I {font-family: MJXc-TeX-main-I,MJXc-TeX-main-Ix,MJXc-TeX-main-Iw}
.MJXc-TeX-main-R {font-family: MJXc-TeX-main-R,MJXc-TeX-main-Rw}
.MJXc-TeX-math-I {font-family: MJXc-TeX-math-I,MJXc-TeX-math-Ix,MJXc-TeX-math-Iw}
.MJXc-TeX-size1-R {font-family: MJXc-TeX-size1-R,MJXc-TeX-size1-Rw}
.MJXc-TeX-size2-R {font-family: MJXc-TeX-size2-R,MJXc-TeX-size2-Rw}
.MJXc-TeX-size3-R {font-family: MJXc-TeX-size3-R,MJXc-TeX-size3-Rw}
.MJXc-TeX-size4-R {font-family: MJXc-TeX-size4-R,MJXc-TeX-size4-Rw}
.MJXc-TeX-vec-R {font-family: MJXc-TeX-vec-R,MJXc-TeX-vec-Rw}
.MJXc-TeX-vec-B {font-family: MJXc-TeX-vec-B,MJXc-TeX-vec-Bx,MJXc-TeX-vec-Bw}
@font-face {font-family: MJXc-TeX-ams-R; src: local('MathJax_AMS'), local('MathJax_AMS-Regular')}
@font-face {font-family: MJXc-TeX-ams-Rw; src /*1*/: url('https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/fonts/HTML-CSS/TeX/eot/MathJax_AMS-Regular.eot'); src /*2*/: url('https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/fonts/HTML-CSS/TeX/woff/MathJax_AMS-Regular.woff') format('woff'), url('https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/fonts/HTML-CSS/TeX/otf/MathJax_AMS-Regular.otf') format('opentype')}
@font-face {font-family: MJXc-TeX-cal-B; src: local('MathJax_Caligraphic Bold'), local('MathJax_Caligraphic-Bold')}
@font-face {font-family: MJXc-TeX-cal-Bx; src: local('MathJax_Caligraphic'); font-weight: bold}
@font-face {font-family: MJXc-TeX-cal-Bw; src /*1*/: url('https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/fonts/HTML-CSS/TeX/eot/MathJax_Caligraphic-Bold.eot'); src /*2*/: url('https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/fonts/HTML-CSS/TeX/woff/MathJax_Caligraphic-Bold.woff') format('woff'), url('https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/fonts/HTML-CSS/TeX/otf/MathJax_Caligraphic-Bold.otf') format('opentype')}
@font-face {font-family: MJXc-TeX-frak-R; src: local('MathJax_Fraktur'), local('MathJax_Fraktur-Regular')}
@font-face {font-family: MJXc-TeX-frak-Rw; src /*1*/: url('https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/fonts/HTML-CSS/TeX/eot/MathJax_Fraktur-Regular.eot'); src /*2*/: url('https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/fonts/HTML-CSS/TeX/woff/MathJax_Fraktur-Regular.woff') format('woff'), url('https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/fonts/HTML-CSS/TeX/otf/MathJax_Fraktur-Regular.otf') format('opentype')}
@font-face {font-family: MJXc-TeX-frak-B; src: local('MathJax_Fraktur Bold'), local('MathJax_Fraktur-Bold')}
@font-face {font-family: MJXc-TeX-frak-Bx; src: local('MathJax_Fraktur'); font-weight: bold}
@font-face {font-family: MJXc-TeX-frak-Bw; src /*1*/: url('https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/fonts/HTML-CSS/TeX/eot/MathJax_Fraktur-Bold.eot'); src /*2*/: url('https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/fonts/HTML-CSS/TeX/woff/MathJax_Fraktur-Bold.woff') format('woff'), url('https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/fonts/HTML-CSS/TeX/otf/MathJax_Fraktur-Bold.otf') format('opentype')}
@font-face {font-family: MJXc-TeX-math-BI; src: local('MathJax_Math BoldItalic'), local('MathJax_Math-BoldItalic')}
@font-face {font-family: MJXc-TeX-math-BIx; src: local('MathJax_Math'); font-weight: bold; font-style: italic}
@font-face {font-family: MJXc-TeX-math-BIw; src /*1*/: url('https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/fonts/HTML-CSS/TeX/eot/MathJax_Math-BoldItalic.eot'); src /*2*/: url('https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/fonts/HTML-CSS/TeX/woff/MathJax_Math-BoldItalic.woff') format('woff'), url('https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/fonts/HTML-CSS/TeX/otf/MathJax_Math-BoldItalic.otf') format('opentype')}
@font-face {font-family: MJXc-TeX-sans-R; src: local('MathJax_SansSerif'), local('MathJax_SansSerif-Regular')}
@font-face {font-family: MJXc-TeX-sans-Rw; src /*1*/: url('https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/fonts/HTML-CSS/TeX/eot/MathJax_SansSerif-Regular.eot'); src /*2*/: url('https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/fonts/HTML-CSS/TeX/woff/MathJax_SansSerif-Regular.woff') format('woff'), url('https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/fonts/HTML-CSS/TeX/otf/MathJax_SansSerif-Regular.otf') format('opentype')}
@font-face {font-family: MJXc-TeX-sans-B; src: local('MathJax_SansSerif Bold'), local('MathJax_SansSerif-Bold')}
@font-face {font-family: MJXc-TeX-sans-Bx; src: local('MathJax_SansSerif'); font-weight: bold}
@font-face {font-family: MJXc-TeX-sans-Bw; src /*1*/: url('https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/fonts/HTML-CSS/TeX/eot/MathJax_SansSerif-Bold.eot'); src /*2*/: url('https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/fonts/HTML-CSS/TeX/woff/MathJax_SansSerif-Bold.woff') format('woff'), url('https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/fonts/HTML-CSS/TeX/otf/MathJax_SansSerif-Bold.otf') format('opentype')}
@font-face {font-family: MJXc-TeX-sans-I; src: local('MathJax_SansSerif Italic'), local('MathJax_SansSerif-Italic')}
@font-face {font-family: MJXc-TeX-sans-Ix; src: local('MathJax_SansSerif'); font-style: italic}
@font-face {font-family: MJXc-TeX-sans-Iw; src /*1*/: url('https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/fonts/HTML-CSS/TeX/eot/MathJax_SansSerif-Italic.eot'); src /*2*/: url('https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/fonts/HTML-CSS/TeX/woff/MathJax_SansSerif-Italic.woff') format('woff'), url('https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/fonts/HTML-CSS/TeX/otf/MathJax_SansSerif-Italic.otf') format('opentype')}
@font-face {font-family: MJXc-TeX-script-R; src: local('MathJax_Script'), local('MathJax_Script-Regular')}
@font-face {font-family: MJXc-TeX-script-Rw; src /*1*/: url('https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/fonts/HTML-CSS/TeX/eot/MathJax_Script-Regular.eot'); src /*2*/: url('https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/fonts/HTML-CSS/TeX/woff/MathJax_Script-Regular.woff') format('woff'), url('https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/fonts/HTML-CSS/TeX/otf/MathJax_Script-Regular.otf') format('opentype')}
@font-face {font-family: MJXc-TeX-type-R; src: local('MathJax_Typewriter'), local('MathJax_Typewriter-Regular')}
@font-face {font-family: MJXc-TeX-type-Rw; src /*1*/: url('https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/fonts/HTML-CSS/TeX/eot/MathJax_Typewriter-Regular.eot'); src /*2*/: url('https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/fonts/HTML-CSS/TeX/woff/MathJax_Typewriter-Regular.woff') format('woff'), url('https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/fonts/HTML-CSS/TeX/otf/MathJax_Typewriter-Regular.otf') format('opentype')}
@font-face {font-family: MJXc-TeX-cal-R; src: local('MathJax_Caligraphic'), local('MathJax_Caligraphic-Regular')}
@font-face {font-family: MJXc-TeX-cal-Rw; src /*1*/: url('https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/fonts/HTML-CSS/TeX/eot/MathJax_Caligraphic-Regular.eot'); src /*2*/: url('https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/fonts/HTML-CSS/TeX/woff/MathJax_Caligraphic-Regular.woff') format('woff'), url('https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/fonts/HTML-CSS/TeX/otf/MathJax_Caligraphic-Regular.otf') format('opentype')}
@font-face {font-family: MJXc-TeX-main-B; src: local('MathJax_Main Bold'), local('MathJax_Main-Bold')}
@font-face {font-family: MJXc-TeX-main-Bx; src: local('MathJax_Main'); font-weight: bold}
@font-face {font-family: MJXc-TeX-main-Bw; src /*1*/: url('https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/fonts/HTML-CSS/TeX/eot/MathJax_Main-Bold.eot'); src /*2*/: url('https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/fonts/HTML-CSS/TeX/woff/MathJax_Main-Bold.woff') format('woff'), url('https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/fonts/HTML-CSS/TeX/otf/MathJax_Main-Bold.otf') format('opentype')}
@font-face {font-family: MJXc-TeX-main-I; src: local('MathJax_Main Italic'), local('MathJax_Main-Italic')}
@font-face {font-family: MJXc-TeX-main-Ix; src: local('MathJax_Main'); font-style: italic}
@font-face {font-family: MJXc-TeX-main-Iw; src /*1*/: url('https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/fonts/HTML-CSS/TeX/eot/MathJax_Main-Italic.eot'); src /*2*/: url('https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/fonts/HTML-CSS/TeX/woff/MathJax_Main-Italic.woff') format('woff'), url('https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/fonts/HTML-CSS/TeX/otf/MathJax_Main-Italic.otf') format('opentype')}
@font-face {font-family: MJXc-TeX-main-R; src: local('MathJax_Main'), local('MathJax_Main-Regular')}
@font-face {font-family: MJXc-TeX-main-Rw; src /*1*/: url('https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/fonts/HTML-CSS/TeX/eot/MathJax_Main-Regular.eot'); src /*2*/: url('https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/fonts/HTML-CSS/TeX/woff/MathJax_Main-Regular.woff') format('woff'), url('https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/fonts/HTML-CSS/TeX/otf/MathJax_Main-Regular.otf') format('opentype')}
@font-face {font-family: MJXc-TeX-math-I; src: local('MathJax_Math Italic'), local('MathJax_Math-Italic')}
@font-face {font-family: MJXc-TeX-math-Ix; src: local('MathJax_Math'); font-style: italic}
@font-face {font-family: MJXc-TeX-math-Iw; src /*1*/: url('https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/fonts/HTML-CSS/TeX/eot/MathJax_Math-Italic.eot'); src /*2*/: url('https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/fonts/HTML-CSS/TeX/woff/MathJax_Math-Italic.woff') format('woff'), url('https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/fonts/HTML-CSS/TeX/otf/MathJax_Math-Italic.otf') format('opentype')}
@font-face {font-family: MJXc-TeX-size1-R; src: local('MathJax_Size1'), local('MathJax_Size1-Regular')}
@font-face {font-family: MJXc-TeX-size1-Rw; src /*1*/: url('https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/fonts/HTML-CSS/TeX/eot/MathJax_Size1-Regular.eot'); src /*2*/: url('https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/fonts/HTML-CSS/TeX/woff/MathJax_Size1-Regular.woff') format('woff'), url('https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/fonts/HTML-CSS/TeX/otf/MathJax_Size1-Regular.otf') format('opentype')}
@font-face {font-family: MJXc-TeX-size2-R; src: local('MathJax_Size2'), local('MathJax_Size2-Regular')}
@font-face {font-family: MJXc-TeX-size2-Rw; src /*1*/: url('https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/fonts/HTML-CSS/TeX/eot/MathJax_Size2-Regular.eot'); src /*2*/: url('https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/fonts/HTML-CSS/TeX/woff/MathJax_Size2-Regular.woff') format('woff'), url('https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/fonts/HTML-CSS/TeX/otf/MathJax_Size2-Regular.otf') format('opentype')}
@font-face {font-family: MJXc-TeX-size3-R; src: local('MathJax_Size3'), local('MathJax_Size3-Regular')}
@font-face {font-family: MJXc-TeX-size3-Rw; src /*1*/: url('https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/fonts/HTML-CSS/TeX/eot/MathJax_Size3-Regular.eot'); src /*2*/: url('https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/fonts/HTML-CSS/TeX/woff/MathJax_Size3-Regular.woff') format('woff'), url('https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/fonts/HTML-CSS/TeX/otf/MathJax_Size3-Regular.otf') format('opentype')}
@font-face {font-family: MJXc-TeX-size4-R; src: local('MathJax_Size4'), local('MathJax_Size4-Regular')}
@font-face {font-family: MJXc-TeX-size4-Rw; src /*1*/: url('https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/fonts/HTML-CSS/TeX/eot/MathJax_Size4-Regular.eot'); src /*2*/: url('https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/fonts/HTML-CSS/TeX/woff/MathJax_Size4-Regular.woff') format('woff'), url('https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/fonts/HTML-CSS/TeX/otf/MathJax_Size4-Regular.otf') format('opentype')}
@font-face {font-family: MJXc-TeX-vec-R; src: local('MathJax_Vector'), local('MathJax_Vector-Regular')}
@font-face {font-family: MJXc-TeX-vec-Rw; src /*1*/: url('https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/fonts/HTML-CSS/TeX/eot/MathJax_Vector-Regular.eot'); src /*2*/: url('https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/fonts/HTML-CSS/TeX/woff/MathJax_Vector-Regular.woff') format('woff'), url('https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/fonts/HTML-CSS/TeX/otf/MathJax_Vector-Regular.otf') format('opentype')}
@font-face {font-family: MJXc-TeX-vec-B; src: local('MathJax_Vector Bold'), local('MathJax_Vector-Bold')}
@font-face {font-family: MJXc-TeX-vec-Bx; src: local('MathJax_Vector'); font-weight: bold}
@font-face {font-family: MJXc-TeX-vec-Bw; src /*1*/: url('https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/fonts/HTML-CSS/TeX/eot/MathJax_Vector-Bold.eot'); src /*2*/: url('https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/fonts/HTML-CSS/TeX/woff/MathJax_Vector-Bold.woff') format('woff'), url('https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/fonts/HTML-CSS/TeX/otf/MathJax_Vector-Bold.otf') format('opentype')}
</style></head>
<body class="jp-Notebook" data-jp-theme-light="true" data-jp-theme-name="JupyterLab Light"><div id="MathJax_Message" style="display: none;"></div>

<div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs  ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-python"><pre><span></span><span class="kn">import</span> <span class="nn">numpy</span> <span class="k">as</span> <span class="nn">np</span>
<span class="kn">import</span> <span class="nn">pandas</span> <span class="k">as</span> <span class="nn">pd</span>
<span class="kn">from</span> <span class="nn">pycaret.regression</span> <span class="kn">import</span> <span class="o">*</span>
<span class="kn">from</span> <span class="nn">plotly.subplots</span> <span class="kn">import</span> <span class="n">make_subplots</span>
<span class="kn">import</span> <span class="nn">plotly.graph_objects</span> <span class="k">as</span> <span class="nn">go</span>
</pre></div>

     </div>
</div>
</div>
</div>

</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs  ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-python"><pre><span></span><span class="k">def</span> <span class="nf">Haversine</span><span class="p">(</span><span class="n">lat1</span><span class="p">:</span> <span class="nb">float</span><span class="p">,</span> <span class="n">lon1</span><span class="p">:</span> <span class="nb">float</span><span class="p">,</span> <span class="n">lat2</span><span class="p">:</span> <span class="nb">float</span><span class="p">,</span> <span class="n">lon2</span><span class="p">:</span> <span class="nb">float</span><span class="p">)</span><span class="o">-&gt;</span> <span class="nb">float</span><span class="p">:</span>
    <span class="n">lon1</span><span class="p">,</span> <span class="n">lon2</span><span class="p">,</span> <span class="n">lat1</span><span class="p">,</span> <span class="n">lat2</span> <span class="o">=</span> <span class="nb">map</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">radians</span><span class="p">,</span> <span class="p">[</span><span class="n">lon1</span><span class="p">,</span> <span class="n">lon2</span><span class="p">,</span> <span class="n">lat1</span><span class="p">,</span> <span class="n">lat2</span><span class="p">])</span>
    <span class="n">diffLon</span> <span class="o">=</span> <span class="n">lon2</span> <span class="o">-</span> <span class="n">lon1</span>
    <span class="n">diffLat</span> <span class="o">=</span> <span class="n">lat2</span> <span class="o">-</span> <span class="n">lat1</span>
    <span class="n">Distance</span> <span class="o">=</span> <span class="mi">2</span> <span class="o">*</span> <span class="mi">6371</span> <span class="o">*</span> <span class="n">np</span><span class="o">.</span><span class="n">arcsin</span><span class="p">(</span>
        <span class="n">np</span><span class="o">.</span><span class="n">sqrt</span><span class="p">(</span>
            <span class="n">np</span><span class="o">.</span><span class="n">sin</span><span class="p">(</span><span class="n">diffLat</span><span class="o">/</span><span class="mi">2</span><span class="p">)</span><span class="o">**</span><span class="mi">2</span> <span class="o">+</span>
            <span class="n">np</span><span class="o">.</span><span class="n">cos</span><span class="p">(</span><span class="n">lat1</span><span class="p">)</span><span class="o">*</span><span class="n">np</span><span class="o">.</span><span class="n">cos</span><span class="p">(</span><span class="n">lat2</span><span class="p">)</span><span class="o">*</span>
            <span class="n">np</span><span class="o">.</span><span class="n">sin</span><span class="p">(</span><span class="n">diffLon</span><span class="o">/</span><span class="mi">2</span><span class="p">)</span><span class="o">**</span><span class="mi">2</span>
        <span class="p">)</span>
    <span class="p">)</span>
    <span class="k">return</span> <span class="n">Distance</span>
</pre></div>

     </div>
</div>
</div>
</div>

</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs  ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-python"><pre><span></span><span class="n">dfUber</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">read_csv</span><span class="p">(</span><span class="s1">'uber.csv'</span><span class="p">)</span>
</pre></div>

     </div>
</div>
</div>
</div>

</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs  ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-python"><pre><span></span><span class="n">dfUber</span><span class="p">[</span><span class="s1">'Distance'</span><span class="p">]</span> <span class="o">=</span> <span class="n">Haversine</span><span class="p">(</span>
    <span class="n">dfUber</span><span class="p">[</span><span class="s1">'pickup_latitude'</span><span class="p">],</span>
    <span class="n">dfUber</span><span class="p">[</span><span class="s1">'pickup_longitude'</span><span class="p">],</span>
    <span class="n">dfUber</span><span class="p">[</span><span class="s1">'dropoff_latitude'</span><span class="p">],</span>
    <span class="n">dfUber</span><span class="p">[</span><span class="s1">'dropoff_longitude'</span><span class="p">]</span>
    <span class="p">)</span>
</pre></div>

     </div>
</div>
</div>
</div>

</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs  ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-python"><pre><span></span><span class="k">del</span> <span class="n">dfUber</span><span class="p">[</span><span class="s1">'pickup_latitude'</span><span class="p">]</span>
<span class="k">del</span> <span class="n">dfUber</span><span class="p">[</span><span class="s1">'pickup_longitude'</span><span class="p">]</span>
<span class="k">del</span> <span class="n">dfUber</span><span class="p">[</span><span class="s1">'dropoff_latitude'</span><span class="p">]</span>
<span class="k">del</span> <span class="n">dfUber</span><span class="p">[</span><span class="s1">'dropoff_longitude'</span><span class="p">]</span>
</pre></div>

     </div>
</div>
</div>
</div>

</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs  ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-python"><pre><span></span><span class="n">dfUber</span> <span class="o">=</span> <span class="n">dfUber</span><span class="p">[</span><span class="n">dfUber</span><span class="p">[</span><span class="s1">'fare_amount'</span><span class="p">]</span> <span class="o">&gt;=</span> <span class="mi">1</span><span class="p">]</span>
<span class="n">dfUber</span> <span class="o">=</span> <span class="n">dfUber</span><span class="p">[</span><span class="n">dfUber</span><span class="p">[</span><span class="s1">'fare_amount'</span><span class="p">]</span> <span class="o">&lt;=</span><span class="mi">80</span><span class="p">]</span>
</pre></div>

     </div>
</div>
</div>
</div>

</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs  ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-python"><pre><span></span><span class="n">dfUber</span> <span class="o">=</span> <span class="n">dfUber</span><span class="p">[</span><span class="n">dfUber</span><span class="p">[</span><span class="s1">'Distance'</span><span class="p">]</span> <span class="o">&lt;=</span> <span class="mi">50</span><span class="p">]</span>
<span class="n">dfUber</span> <span class="o">=</span> <span class="n">dfUber</span><span class="p">[</span><span class="n">dfUber</span><span class="p">[</span><span class="s1">'Distance'</span><span class="p">]</span> <span class="o">&gt;=</span> <span class="mf">0.1</span><span class="p">]</span>
</pre></div>

     </div>
</div>
</div>
</div>

</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs  ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-python"><pre><span></span><span class="n">dfUber</span> <span class="o">=</span> <span class="n">dfUber</span><span class="p">[</span><span class="n">dfUber</span><span class="p">[</span><span class="s1">'passenger_count'</span><span class="p">]</span> <span class="o">&gt;=</span> <span class="mi">1</span><span class="p">]</span>
<span class="n">dfUber</span> <span class="o">=</span> <span class="n">dfUber</span><span class="p">[</span><span class="n">dfUber</span><span class="p">[</span><span class="s1">'passenger_count'</span><span class="p">]</span> <span class="o">&lt;=</span> <span class="mi">4</span><span class="p">]</span>
</pre></div>

     </div>
</div>
</div>
</div>

</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs  ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-python"><pre><span></span><span class="n">dfUber</span><span class="p">[</span><span class="s1">'pickup_datetime'</span><span class="p">]</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">to_datetime</span><span class="p">(</span><span class="n">dfUber</span><span class="p">[</span><span class="s1">'pickup_datetime'</span><span class="p">]</span> <span class="p">)</span>
<span class="n">dfUber</span><span class="p">[</span><span class="s1">'Hour'</span><span class="p">]</span> <span class="o">=</span> <span class="n">dfUber</span><span class="p">[</span><span class="s1">'pickup_datetime'</span><span class="p">]</span><span class="o">.</span><span class="n">apply</span><span class="p">(</span><span class="k">lambda</span> <span class="n">x</span><span class="p">:</span> <span class="n">x</span><span class="o">.</span><span class="n">hour</span><span class="p">)</span>
<span class="n">dfUber</span><span class="p">[</span><span class="s1">'Minute'</span><span class="p">]</span> <span class="o">=</span> <span class="n">dfUber</span><span class="p">[</span><span class="s1">'pickup_datetime'</span><span class="p">]</span><span class="o">.</span><span class="n">apply</span><span class="p">(</span><span class="k">lambda</span> <span class="n">x</span><span class="p">:</span> <span class="n">x</span><span class="o">.</span><span class="n">minute</span><span class="p">)</span>
<span class="n">dfUber</span><span class="p">[</span><span class="s1">'Day'</span><span class="p">]</span> <span class="o">=</span> <span class="n">dfUber</span><span class="p">[</span><span class="s1">'pickup_datetime'</span><span class="p">]</span><span class="o">.</span><span class="n">apply</span><span class="p">(</span><span class="k">lambda</span> <span class="n">x</span><span class="p">:</span> <span class="n">x</span><span class="o">.</span><span class="n">dayofweek</span><span class="p">)</span>
<span class="k">del</span> <span class="n">dfUber</span><span class="p">[</span><span class="s1">'pickup_datetime'</span><span class="p">]</span>
</pre></div>

     </div>
</div>
</div>
</div>

</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs  ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-python"><pre><span></span><span class="n">fig</span> <span class="o">=</span> <span class="n">make_subplots</span><span class="p">(</span><span class="n">rows</span><span class="o">=</span><span class="mi">3</span><span class="p">,</span> <span class="n">cols</span><span class="o">=</span><span class="mi">1</span><span class="p">)</span>
</pre></div>

     </div>
</div>
</div>
</div>

</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-python"><pre><span></span><span class="n">fig</span><span class="o">.</span><span class="n">add_trace</span><span class="p">(</span>
    <span class="n">go</span><span class="o">.</span><span class="n">Box</span><span class="p">(</span>
        <span class="n">x</span><span class="o">=</span><span class="n">dfUber</span><span class="p">[</span><span class="s1">'Distance'</span><span class="p">],</span>
        <span class="n">name</span><span class="o">=</span><span class="s1">'Distância'</span>
    <span class="p">),</span>
    <span class="n">row</span><span class="o">=</span><span class="mi">1</span><span class="p">,</span>
    <span class="n">col</span><span class="o">=</span><span class="mi">1</span>
<span class="p">)</span>

<span class="n">fig</span><span class="o">.</span><span class="n">add_trace</span><span class="p">(</span>
    <span class="n">go</span><span class="o">.</span><span class="n">Box</span><span class="p">(</span>
        <span class="n">x</span><span class="o">=</span><span class="n">dfUber</span><span class="p">[</span><span class="s1">'fare_amount'</span><span class="p">],</span>
        <span class="n">name</span><span class="o">=</span><span class="s1">'Preço'</span>
    <span class="p">),</span>
    <span class="n">row</span><span class="o">=</span><span class="mi">2</span><span class="p">,</span>
    <span class="n">col</span><span class="o">=</span><span class="mi">1</span>
<span class="p">)</span>

<span class="n">fig</span><span class="o">.</span><span class="n">add_trace</span><span class="p">(</span>
    <span class="n">go</span><span class="o">.</span><span class="n">Bar</span><span class="p">(</span>
        <span class="n">y</span><span class="o">=</span><span class="n">dfUber</span><span class="p">[</span><span class="s1">'passenger_count'</span><span class="p">]</span><span class="o">.</span><span class="n">value_counts</span><span class="p">(),</span>
        <span class="n">name</span><span class="o">=</span><span class="s1">'Quantidade de Passageiros'</span><span class="p">,</span>
        <span class="n">x</span><span class="o">=</span><span class="p">[</span><span class="mi">1</span><span class="p">,</span><span class="mi">2</span><span class="p">,</span><span class="mi">3</span><span class="p">,</span><span class="mi">4</span><span class="p">]</span>
    <span class="p">),</span>
    <span class="n">row</span><span class="o">=</span><span class="mi">3</span><span class="p">,</span>
    <span class="n">col</span><span class="o">=</span><span class="mi">1</span>
<span class="p">)</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>





</div>

</div>

</div>

</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-python"><pre><span></span><span class="n">fig</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>





</div>

</div>

</div>

</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs  ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-python"><pre><span></span><span class="k">del</span> <span class="n">dfUber</span><span class="p">[</span><span class="s1">'Unnamed: 0'</span><span class="p">]</span>
<span class="k">del</span> <span class="n">dfUber</span><span class="p">[</span><span class="s1">'key'</span><span class="p">]</span>
</pre></div>

     </div>
</div>
</div>
</div>

</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-python"><pre><span></span><span class="n">EnvRegr</span> <span class="o">=</span> <span class="n">setup</span><span class="p">(</span><span class="n">dfUber</span><span class="p">,</span> <span class="n">target</span><span class="o">=</span><span class="s1">'fare_amount'</span><span class="p">,</span> <span class="n">normalize</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>



<div class="jp-RenderedHTMLCommon jp-RenderedHTML jp-OutputArea-output " data-mime-type="text/html">
<style type="text/css">
#T_3307a_row27_col1, #T_3307a_row42_col1 {
  background-color: lightgreen;
}
</style>
<table id="T_3307a">
  <thead>
    <tr>
      <th class="blank level0">&nbsp;</th>
      <th id="T_3307a_level0_col0" class="col_heading level0 col0">Description</th>
      <th id="T_3307a_level0_col1" class="col_heading level0 col1">Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_3307a_level0_row0" class="row_heading level0 row0">0</th>
      <td id="T_3307a_row0_col0" class="data row0 col0">session_id</td>
      <td id="T_3307a_row0_col1" class="data row0 col1">4673</td>
    </tr>
    <tr>
      <th id="T_3307a_level0_row1" class="row_heading level0 row1">1</th>
      <td id="T_3307a_row1_col0" class="data row1 col0">Target</td>
      <td id="T_3307a_row1_col1" class="data row1 col1">fare_amount</td>
    </tr>
    <tr>
      <th id="T_3307a_level0_row2" class="row_heading level0 row2">2</th>
      <td id="T_3307a_row2_col0" class="data row2 col0">Original Data</td>
      <td id="T_3307a_row2_col1" class="data row2 col1">(174233, 6)</td>
    </tr>
    <tr>
      <th id="T_3307a_level0_row3" class="row_heading level0 row3">3</th>
      <td id="T_3307a_row3_col0" class="data row3 col0">Missing Values</td>
      <td id="T_3307a_row3_col1" class="data row3 col1">False</td>
    </tr>
    <tr>
      <th id="T_3307a_level0_row4" class="row_heading level0 row4">4</th>
      <td id="T_3307a_row4_col0" class="data row4 col0">Numeric Features</td>
      <td id="T_3307a_row4_col1" class="data row4 col1">3</td>
    </tr>
    <tr>
      <th id="T_3307a_level0_row5" class="row_heading level0 row5">5</th>
      <td id="T_3307a_row5_col0" class="data row5 col0">Categorical Features</td>
      <td id="T_3307a_row5_col1" class="data row5 col1">2</td>
    </tr>
    <tr>
      <th id="T_3307a_level0_row6" class="row_heading level0 row6">6</th>
      <td id="T_3307a_row6_col0" class="data row6 col0">Ordinal Features</td>
      <td id="T_3307a_row6_col1" class="data row6 col1">False</td>
    </tr>
    <tr>
      <th id="T_3307a_level0_row7" class="row_heading level0 row7">7</th>
      <td id="T_3307a_row7_col0" class="data row7 col0">High Cardinality Features</td>
      <td id="T_3307a_row7_col1" class="data row7 col1">False</td>
    </tr>
    <tr>
      <th id="T_3307a_level0_row8" class="row_heading level0 row8">8</th>
      <td id="T_3307a_row8_col0" class="data row8 col0">High Cardinality Method</td>
      <td id="T_3307a_row8_col1" class="data row8 col1">None</td>
    </tr>
    <tr>
      <th id="T_3307a_level0_row9" class="row_heading level0 row9">9</th>
      <td id="T_3307a_row9_col0" class="data row9 col0">Transformed Train Set</td>
      <td id="T_3307a_row9_col1" class="data row9 col1">(121963, 14)</td>
    </tr>
    <tr>
      <th id="T_3307a_level0_row10" class="row_heading level0 row10">10</th>
      <td id="T_3307a_row10_col0" class="data row10 col0">Transformed Test Set</td>
      <td id="T_3307a_row10_col1" class="data row10 col1">(52270, 14)</td>
    </tr>
    <tr>
      <th id="T_3307a_level0_row11" class="row_heading level0 row11">11</th>
      <td id="T_3307a_row11_col0" class="data row11 col0">Shuffle Train-Test</td>
      <td id="T_3307a_row11_col1" class="data row11 col1">True</td>
    </tr>
    <tr>
      <th id="T_3307a_level0_row12" class="row_heading level0 row12">12</th>
      <td id="T_3307a_row12_col0" class="data row12 col0">Stratify Train-Test</td>
      <td id="T_3307a_row12_col1" class="data row12 col1">False</td>
    </tr>
    <tr>
      <th id="T_3307a_level0_row13" class="row_heading level0 row13">13</th>
      <td id="T_3307a_row13_col0" class="data row13 col0">Fold Generator</td>
      <td id="T_3307a_row13_col1" class="data row13 col1">KFold</td>
    </tr>
    <tr>
      <th id="T_3307a_level0_row14" class="row_heading level0 row14">14</th>
      <td id="T_3307a_row14_col0" class="data row14 col0">Fold Number</td>
      <td id="T_3307a_row14_col1" class="data row14 col1">10</td>
    </tr>
    <tr>
      <th id="T_3307a_level0_row15" class="row_heading level0 row15">15</th>
      <td id="T_3307a_row15_col0" class="data row15 col0">CPU Jobs</td>
      <td id="T_3307a_row15_col1" class="data row15 col1">-1</td>
    </tr>
    <tr>
      <th id="T_3307a_level0_row16" class="row_heading level0 row16">16</th>
      <td id="T_3307a_row16_col0" class="data row16 col0">Use GPU</td>
      <td id="T_3307a_row16_col1" class="data row16 col1">False</td>
    </tr>
    <tr>
      <th id="T_3307a_level0_row17" class="row_heading level0 row17">17</th>
      <td id="T_3307a_row17_col0" class="data row17 col0">Log Experiment</td>
      <td id="T_3307a_row17_col1" class="data row17 col1">False</td>
    </tr>
    <tr>
      <th id="T_3307a_level0_row18" class="row_heading level0 row18">18</th>
      <td id="T_3307a_row18_col0" class="data row18 col0">Experiment Name</td>
      <td id="T_3307a_row18_col1" class="data row18 col1">reg-default-name</td>
    </tr>
    <tr>
      <th id="T_3307a_level0_row19" class="row_heading level0 row19">19</th>
      <td id="T_3307a_row19_col0" class="data row19 col0">USI</td>
      <td id="T_3307a_row19_col1" class="data row19 col1">d7a2</td>
    </tr>
    <tr>
      <th id="T_3307a_level0_row20" class="row_heading level0 row20">20</th>
      <td id="T_3307a_row20_col0" class="data row20 col0">Imputation Type</td>
      <td id="T_3307a_row20_col1" class="data row20 col1">simple</td>
    </tr>
    <tr>
      <th id="T_3307a_level0_row21" class="row_heading level0 row21">21</th>
      <td id="T_3307a_row21_col0" class="data row21 col0">Iterative Imputation Iteration</td>
      <td id="T_3307a_row21_col1" class="data row21 col1">None</td>
    </tr>
    <tr>
      <th id="T_3307a_level0_row22" class="row_heading level0 row22">22</th>
      <td id="T_3307a_row22_col0" class="data row22 col0">Numeric Imputer</td>
      <td id="T_3307a_row22_col1" class="data row22 col1">mean</td>
    </tr>
    <tr>
      <th id="T_3307a_level0_row23" class="row_heading level0 row23">23</th>
      <td id="T_3307a_row23_col0" class="data row23 col0">Iterative Imputation Numeric Model</td>
      <td id="T_3307a_row23_col1" class="data row23 col1">None</td>
    </tr>
    <tr>
      <th id="T_3307a_level0_row24" class="row_heading level0 row24">24</th>
      <td id="T_3307a_row24_col0" class="data row24 col0">Categorical Imputer</td>
      <td id="T_3307a_row24_col1" class="data row24 col1">constant</td>
    </tr>
    <tr>
      <th id="T_3307a_level0_row25" class="row_heading level0 row25">25</th>
      <td id="T_3307a_row25_col0" class="data row25 col0">Iterative Imputation Categorical Model</td>
      <td id="T_3307a_row25_col1" class="data row25 col1">None</td>
    </tr>
    <tr>
      <th id="T_3307a_level0_row26" class="row_heading level0 row26">26</th>
      <td id="T_3307a_row26_col0" class="data row26 col0">Unknown Categoricals Handling</td>
      <td id="T_3307a_row26_col1" class="data row26 col1">least_frequent</td>
    </tr>
    <tr>
      <th id="T_3307a_level0_row27" class="row_heading level0 row27">27</th>
      <td id="T_3307a_row27_col0" class="data row27 col0">Normalize</td>
      <td id="T_3307a_row27_col1" class="data row27 col1">True</td>
    </tr>
    <tr>
      <th id="T_3307a_level0_row28" class="row_heading level0 row28">28</th>
      <td id="T_3307a_row28_col0" class="data row28 col0">Normalize Method</td>
      <td id="T_3307a_row28_col1" class="data row28 col1">zscore</td>
    </tr>
    <tr>
      <th id="T_3307a_level0_row29" class="row_heading level0 row29">29</th>
      <td id="T_3307a_row29_col0" class="data row29 col0">Transformation</td>
      <td id="T_3307a_row29_col1" class="data row29 col1">False</td>
    </tr>
    <tr>
      <th id="T_3307a_level0_row30" class="row_heading level0 row30">30</th>
      <td id="T_3307a_row30_col0" class="data row30 col0">Transformation Method</td>
      <td id="T_3307a_row30_col1" class="data row30 col1">None</td>
    </tr>
    <tr>
      <th id="T_3307a_level0_row31" class="row_heading level0 row31">31</th>
      <td id="T_3307a_row31_col0" class="data row31 col0">PCA</td>
      <td id="T_3307a_row31_col1" class="data row31 col1">False</td>
    </tr>
    <tr>
      <th id="T_3307a_level0_row32" class="row_heading level0 row32">32</th>
      <td id="T_3307a_row32_col0" class="data row32 col0">PCA Method</td>
      <td id="T_3307a_row32_col1" class="data row32 col1">None</td>
    </tr>
    <tr>
      <th id="T_3307a_level0_row33" class="row_heading level0 row33">33</th>
      <td id="T_3307a_row33_col0" class="data row33 col0">PCA Components</td>
      <td id="T_3307a_row33_col1" class="data row33 col1">None</td>
    </tr>
    <tr>
      <th id="T_3307a_level0_row34" class="row_heading level0 row34">34</th>
      <td id="T_3307a_row34_col0" class="data row34 col0">Ignore Low Variance</td>
      <td id="T_3307a_row34_col1" class="data row34 col1">False</td>
    </tr>
    <tr>
      <th id="T_3307a_level0_row35" class="row_heading level0 row35">35</th>
      <td id="T_3307a_row35_col0" class="data row35 col0">Combine Rare Levels</td>
      <td id="T_3307a_row35_col1" class="data row35 col1">False</td>
    </tr>
    <tr>
      <th id="T_3307a_level0_row36" class="row_heading level0 row36">36</th>
      <td id="T_3307a_row36_col0" class="data row36 col0">Rare Level Threshold</td>
      <td id="T_3307a_row36_col1" class="data row36 col1">None</td>
    </tr>
    <tr>
      <th id="T_3307a_level0_row37" class="row_heading level0 row37">37</th>
      <td id="T_3307a_row37_col0" class="data row37 col0">Numeric Binning</td>
      <td id="T_3307a_row37_col1" class="data row37 col1">False</td>
    </tr>
    <tr>
      <th id="T_3307a_level0_row38" class="row_heading level0 row38">38</th>
      <td id="T_3307a_row38_col0" class="data row38 col0">Remove Outliers</td>
      <td id="T_3307a_row38_col1" class="data row38 col1">False</td>
    </tr>
    <tr>
      <th id="T_3307a_level0_row39" class="row_heading level0 row39">39</th>
      <td id="T_3307a_row39_col0" class="data row39 col0">Outliers Threshold</td>
      <td id="T_3307a_row39_col1" class="data row39 col1">None</td>
    </tr>
    <tr>
      <th id="T_3307a_level0_row40" class="row_heading level0 row40">40</th>
      <td id="T_3307a_row40_col0" class="data row40 col0">Remove Multicollinearity</td>
      <td id="T_3307a_row40_col1" class="data row40 col1">False</td>
    </tr>
    <tr>
      <th id="T_3307a_level0_row41" class="row_heading level0 row41">41</th>
      <td id="T_3307a_row41_col0" class="data row41 col0">Multicollinearity Threshold</td>
      <td id="T_3307a_row41_col1" class="data row41 col1">None</td>
    </tr>
    <tr>
      <th id="T_3307a_level0_row42" class="row_heading level0 row42">42</th>
      <td id="T_3307a_row42_col0" class="data row42 col0">Remove Perfect Collinearity</td>
      <td id="T_3307a_row42_col1" class="data row42 col1">True</td>
    </tr>
    <tr>
      <th id="T_3307a_level0_row43" class="row_heading level0 row43">43</th>
      <td id="T_3307a_row43_col0" class="data row43 col0">Clustering</td>
      <td id="T_3307a_row43_col1" class="data row43 col1">False</td>
    </tr>
    <tr>
      <th id="T_3307a_level0_row44" class="row_heading level0 row44">44</th>
      <td id="T_3307a_row44_col0" class="data row44 col0">Clustering Iteration</td>
      <td id="T_3307a_row44_col1" class="data row44 col1">None</td>
    </tr>
    <tr>
      <th id="T_3307a_level0_row45" class="row_heading level0 row45">45</th>
      <td id="T_3307a_row45_col0" class="data row45 col0">Polynomial Features</td>
      <td id="T_3307a_row45_col1" class="data row45 col1">False</td>
    </tr>
    <tr>
      <th id="T_3307a_level0_row46" class="row_heading level0 row46">46</th>
      <td id="T_3307a_row46_col0" class="data row46 col0">Polynomial Degree</td>
      <td id="T_3307a_row46_col1" class="data row46 col1">None</td>
    </tr>
    <tr>
      <th id="T_3307a_level0_row47" class="row_heading level0 row47">47</th>
      <td id="T_3307a_row47_col0" class="data row47 col0">Trignometry Features</td>
      <td id="T_3307a_row47_col1" class="data row47 col1">False</td>
    </tr>
    <tr>
      <th id="T_3307a_level0_row48" class="row_heading level0 row48">48</th>
      <td id="T_3307a_row48_col0" class="data row48 col0">Polynomial Threshold</td>
      <td id="T_3307a_row48_col1" class="data row48 col1">None</td>
    </tr>
    <tr>
      <th id="T_3307a_level0_row49" class="row_heading level0 row49">49</th>
      <td id="T_3307a_row49_col0" class="data row49 col0">Group Features</td>
      <td id="T_3307a_row49_col1" class="data row49 col1">False</td>
    </tr>
    <tr>
      <th id="T_3307a_level0_row50" class="row_heading level0 row50">50</th>
      <td id="T_3307a_row50_col0" class="data row50 col0">Feature Selection</td>
      <td id="T_3307a_row50_col1" class="data row50 col1">False</td>
    </tr>
    <tr>
      <th id="T_3307a_level0_row51" class="row_heading level0 row51">51</th>
      <td id="T_3307a_row51_col0" class="data row51 col0">Feature Selection Method</td>
      <td id="T_3307a_row51_col1" class="data row51 col1">classic</td>
    </tr>
    <tr>
      <th id="T_3307a_level0_row52" class="row_heading level0 row52">52</th>
      <td id="T_3307a_row52_col0" class="data row52 col0">Features Selection Threshold</td>
      <td id="T_3307a_row52_col1" class="data row52 col1">None</td>
    </tr>
    <tr>
      <th id="T_3307a_level0_row53" class="row_heading level0 row53">53</th>
      <td id="T_3307a_row53_col0" class="data row53 col0">Feature Interaction</td>
      <td id="T_3307a_row53_col1" class="data row53 col1">False</td>
    </tr>
    <tr>
      <th id="T_3307a_level0_row54" class="row_heading level0 row54">54</th>
      <td id="T_3307a_row54_col0" class="data row54 col0">Feature Ratio</td>
      <td id="T_3307a_row54_col1" class="data row54 col1">False</td>
    </tr>
    <tr>
      <th id="T_3307a_level0_row55" class="row_heading level0 row55">55</th>
      <td id="T_3307a_row55_col0" class="data row55 col0">Interaction Threshold</td>
      <td id="T_3307a_row55_col1" class="data row55 col1">None</td>
    </tr>
    <tr>
      <th id="T_3307a_level0_row56" class="row_heading level0 row56">56</th>
      <td id="T_3307a_row56_col0" class="data row56 col0">Transform Target</td>
      <td id="T_3307a_row56_col1" class="data row56 col1">False</td>
    </tr>
    <tr>
      <th id="T_3307a_level0_row57" class="row_heading level0 row57">57</th>
      <td id="T_3307a_row57_col0" class="data row57 col0">Transform Target Method</td>
      <td id="T_3307a_row57_col1" class="data row57 col1">box-cox</td>
    </tr>
  </tbody>
</table>

</div>

</div>

</div>

</div>

</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-python"><pre><span></span><span class="n">best</span> <span class="o">=</span> <span class="n">create_model</span><span class="p">(</span><span class="s1">'lightgbm'</span><span class="p">)</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>



<div class="jp-RenderedHTMLCommon jp-RenderedHTML jp-OutputArea-output " data-mime-type="text/html">
<style type="text/css">
#T_4561c_row10_col0, #T_4561c_row10_col1, #T_4561c_row10_col2, #T_4561c_row10_col3, #T_4561c_row10_col4, #T_4561c_row10_col5 {
  background: yellow;
}
</style>
<table id="T_4561c">
  <thead>
    <tr>
      <th class="blank level0">&nbsp;</th>
      <th id="T_4561c_level0_col0" class="col_heading level0 col0">MAE</th>
      <th id="T_4561c_level0_col1" class="col_heading level0 col1">MSE</th>
      <th id="T_4561c_level0_col2" class="col_heading level0 col2">RMSE</th>
      <th id="T_4561c_level0_col3" class="col_heading level0 col3">R2</th>
      <th id="T_4561c_level0_col4" class="col_heading level0 col4">RMSLE</th>
      <th id="T_4561c_level0_col5" class="col_heading level0 col5">MAPE</th>
    </tr>
    <tr>
      <th class="index_name level0">Fold</th>
      <th class="blank col0">&nbsp;</th>
      <th class="blank col1">&nbsp;</th>
      <th class="blank col2">&nbsp;</th>
      <th class="blank col3">&nbsp;</th>
      <th class="blank col4">&nbsp;</th>
      <th class="blank col5">&nbsp;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_4561c_level0_row0" class="row_heading level0 row0">0</th>
      <td id="T_4561c_row0_col0" class="data row0 col0">2.1013</td>
      <td id="T_4561c_row0_col1" class="data row0 col1">12.4597</td>
      <td id="T_4561c_row0_col2" class="data row0 col2">3.5298</td>
      <td id="T_4561c_row0_col3" class="data row0 col3">0.8440</td>
      <td id="T_4561c_row0_col4" class="data row0 col4">0.2313</td>
      <td id="T_4561c_row0_col5" class="data row0 col5">0.2036</td>
    </tr>
    <tr>
      <th id="T_4561c_level0_row1" class="row_heading level0 row1">1</th>
      <td id="T_4561c_row1_col0" class="data row1 col0">2.0870</td>
      <td id="T_4561c_row1_col1" class="data row1 col1">14.0089</td>
      <td id="T_4561c_row1_col2" class="data row1 col2">3.7429</td>
      <td id="T_4561c_row1_col3" class="data row1 col3">0.8246</td>
      <td id="T_4561c_row1_col4" class="data row1 col4">0.2371</td>
      <td id="T_4561c_row1_col5" class="data row1 col5">0.2039</td>
    </tr>
    <tr>
      <th id="T_4561c_level0_row2" class="row_heading level0 row2">2</th>
      <td id="T_4561c_row2_col0" class="data row2 col0">2.1102</td>
      <td id="T_4561c_row2_col1" class="data row2 col1">14.9352</td>
      <td id="T_4561c_row2_col2" class="data row2 col2">3.8646</td>
      <td id="T_4561c_row2_col3" class="data row2 col3">0.8258</td>
      <td id="T_4561c_row2_col4" class="data row2 col4">0.2311</td>
      <td id="T_4561c_row2_col5" class="data row2 col5">0.1988</td>
    </tr>
    <tr>
      <th id="T_4561c_level0_row3" class="row_heading level0 row3">3</th>
      <td id="T_4561c_row3_col0" class="data row3 col0">2.0565</td>
      <td id="T_4561c_row3_col1" class="data row3 col1">12.9266</td>
      <td id="T_4561c_row3_col2" class="data row3 col2">3.5954</td>
      <td id="T_4561c_row3_col3" class="data row3 col3">0.8415</td>
      <td id="T_4561c_row3_col4" class="data row3 col4">0.2301</td>
      <td id="T_4561c_row3_col5" class="data row3 col5">0.1994</td>
    </tr>
    <tr>
      <th id="T_4561c_level0_row4" class="row_heading level0 row4">4</th>
      <td id="T_4561c_row4_col0" class="data row4 col0">2.1495</td>
      <td id="T_4561c_row4_col1" class="data row4 col1">14.9747</td>
      <td id="T_4561c_row4_col2" class="data row4 col2">3.8697</td>
      <td id="T_4561c_row4_col3" class="data row4 col3">0.8242</td>
      <td id="T_4561c_row4_col4" class="data row4 col4">0.2384</td>
      <td id="T_4561c_row4_col5" class="data row4 col5">0.2036</td>
    </tr>
    <tr>
      <th id="T_4561c_level0_row5" class="row_heading level0 row5">5</th>
      <td id="T_4561c_row5_col0" class="data row5 col0">2.1227</td>
      <td id="T_4561c_row5_col1" class="data row5 col1">14.3503</td>
      <td id="T_4561c_row5_col2" class="data row5 col2">3.7882</td>
      <td id="T_4561c_row5_col3" class="data row5 col3">0.8341</td>
      <td id="T_4561c_row5_col4" class="data row5 col4">0.2333</td>
      <td id="T_4561c_row5_col5" class="data row5 col5">0.1991</td>
    </tr>
    <tr>
      <th id="T_4561c_level0_row6" class="row_heading level0 row6">6</th>
      <td id="T_4561c_row6_col0" class="data row6 col0">2.0968</td>
      <td id="T_4561c_row6_col1" class="data row6 col1">14.2362</td>
      <td id="T_4561c_row6_col2" class="data row6 col2">3.7731</td>
      <td id="T_4561c_row6_col3" class="data row6 col3">0.8310</td>
      <td id="T_4561c_row6_col4" class="data row6 col4">0.2318</td>
      <td id="T_4561c_row6_col5" class="data row6 col5">0.1968</td>
    </tr>
    <tr>
      <th id="T_4561c_level0_row7" class="row_heading level0 row7">7</th>
      <td id="T_4561c_row7_col0" class="data row7 col0">2.1424</td>
      <td id="T_4561c_row7_col1" class="data row7 col1">13.4225</td>
      <td id="T_4561c_row7_col2" class="data row7 col2">3.6637</td>
      <td id="T_4561c_row7_col3" class="data row7 col3">0.8394</td>
      <td id="T_4561c_row7_col4" class="data row7 col4">0.2346</td>
      <td id="T_4561c_row7_col5" class="data row7 col5">0.2033</td>
    </tr>
    <tr>
      <th id="T_4561c_level0_row8" class="row_heading level0 row8">8</th>
      <td id="T_4561c_row8_col0" class="data row8 col0">2.1277</td>
      <td id="T_4561c_row8_col1" class="data row8 col1">14.4568</td>
      <td id="T_4561c_row8_col2" class="data row8 col2">3.8022</td>
      <td id="T_4561c_row8_col3" class="data row8 col3">0.8188</td>
      <td id="T_4561c_row8_col4" class="data row8 col4">0.2398</td>
      <td id="T_4561c_row8_col5" class="data row8 col5">0.2072</td>
    </tr>
    <tr>
      <th id="T_4561c_level0_row9" class="row_heading level0 row9">9</th>
      <td id="T_4561c_row9_col0" class="data row9 col0">2.0910</td>
      <td id="T_4561c_row9_col1" class="data row9 col1">13.0607</td>
      <td id="T_4561c_row9_col2" class="data row9 col2">3.6140</td>
      <td id="T_4561c_row9_col3" class="data row9 col3">0.8411</td>
      <td id="T_4561c_row9_col4" class="data row9 col4">0.2315</td>
      <td id="T_4561c_row9_col5" class="data row9 col5">0.2011</td>
    </tr>
    <tr>
      <th id="T_4561c_level0_row10" class="row_heading level0 row10">Mean</th>
      <td id="T_4561c_row10_col0" class="data row10 col0">2.1085</td>
      <td id="T_4561c_row10_col1" class="data row10 col1">13.8832</td>
      <td id="T_4561c_row10_col2" class="data row10 col2">3.7243</td>
      <td id="T_4561c_row10_col3" class="data row10 col3">0.8324</td>
      <td id="T_4561c_row10_col4" class="data row10 col4">0.2339</td>
      <td id="T_4561c_row10_col5" class="data row10 col5">0.2017</td>
    </tr>
    <tr>
      <th id="T_4561c_level0_row11" class="row_heading level0 row11">Std</th>
      <td id="T_4561c_row11_col0" class="data row11 col0">0.0266</td>
      <td id="T_4561c_row11_col1" class="data row11 col1">0.8256</td>
      <td id="T_4561c_row11_col2" class="data row11 col2">0.1114</td>
      <td id="T_4561c_row11_col3" class="data row11 col3">0.0084</td>
      <td id="T_4561c_row11_col4" class="data row11 col4">0.0032</td>
      <td id="T_4561c_row11_col5" class="data row11 col5">0.0030</td>
    </tr>
  </tbody>
</table>

</div>

</div>

</div>

</div>

</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-python"><pre><span></span><span class="n">UberMLTunned</span> <span class="o">=</span> <span class="n">tune_model</span><span class="p">(</span><span class="n">best</span><span class="p">)</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>



<div class="jp-RenderedHTMLCommon jp-RenderedHTML jp-OutputArea-output " data-mime-type="text/html">
<style type="text/css">
#T_01116_row10_col0, #T_01116_row10_col1, #T_01116_row10_col2, #T_01116_row10_col3, #T_01116_row10_col4, #T_01116_row10_col5 {
  background: yellow;
}
</style>
<table id="T_01116">
  <thead>
    <tr>
      <th class="blank level0">&nbsp;</th>
      <th id="T_01116_level0_col0" class="col_heading level0 col0">MAE</th>
      <th id="T_01116_level0_col1" class="col_heading level0 col1">MSE</th>
      <th id="T_01116_level0_col2" class="col_heading level0 col2">RMSE</th>
      <th id="T_01116_level0_col3" class="col_heading level0 col3">R2</th>
      <th id="T_01116_level0_col4" class="col_heading level0 col4">RMSLE</th>
      <th id="T_01116_level0_col5" class="col_heading level0 col5">MAPE</th>
    </tr>
    <tr>
      <th class="index_name level0">Fold</th>
      <th class="blank col0">&nbsp;</th>
      <th class="blank col1">&nbsp;</th>
      <th class="blank col2">&nbsp;</th>
      <th class="blank col3">&nbsp;</th>
      <th class="blank col4">&nbsp;</th>
      <th class="blank col5">&nbsp;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_01116_level0_row0" class="row_heading level0 row0">0</th>
      <td id="T_01116_row0_col0" class="data row0 col0">2.1362</td>
      <td id="T_01116_row0_col1" class="data row0 col1">12.6925</td>
      <td id="T_01116_row0_col2" class="data row0 col2">3.5627</td>
      <td id="T_01116_row0_col3" class="data row0 col3">0.8411</td>
      <td id="T_01116_row0_col4" class="data row0 col4">0.2348</td>
      <td id="T_01116_row0_col5" class="data row0 col5">0.2076</td>
    </tr>
    <tr>
      <th id="T_01116_level0_row1" class="row_heading level0 row1">1</th>
      <td id="T_01116_row1_col0" class="data row1 col0">2.1351</td>
      <td id="T_01116_row1_col1" class="data row1 col1">14.3259</td>
      <td id="T_01116_row1_col2" class="data row1 col2">3.7850</td>
      <td id="T_01116_row1_col3" class="data row1 col3">0.8206</td>
      <td id="T_01116_row1_col4" class="data row1 col4">0.2405</td>
      <td id="T_01116_row1_col5" class="data row1 col5">0.2092</td>
    </tr>
    <tr>
      <th id="T_01116_level0_row2" class="row_heading level0 row2">2</th>
      <td id="T_01116_row2_col0" class="data row2 col0">2.1548</td>
      <td id="T_01116_row2_col1" class="data row2 col1">15.1606</td>
      <td id="T_01116_row2_col2" class="data row2 col2">3.8937</td>
      <td id="T_01116_row2_col3" class="data row2 col3">0.8232</td>
      <td id="T_01116_row2_col4" class="data row2 col4">0.2345</td>
      <td id="T_01116_row2_col5" class="data row2 col5">0.2035</td>
    </tr>
    <tr>
      <th id="T_01116_level0_row3" class="row_heading level0 row3">3</th>
      <td id="T_01116_row3_col0" class="data row3 col0">2.1054</td>
      <td id="T_01116_row3_col1" class="data row3 col1">13.4011</td>
      <td id="T_01116_row3_col2" class="data row3 col2">3.6608</td>
      <td id="T_01116_row3_col3" class="data row3 col3">0.8357</td>
      <td id="T_01116_row3_col4" class="data row3 col4">0.2340</td>
      <td id="T_01116_row3_col5" class="data row3 col5">0.2038</td>
    </tr>
    <tr>
      <th id="T_01116_level0_row4" class="row_heading level0 row4">4</th>
      <td id="T_01116_row4_col0" class="data row4 col0">2.1813</td>
      <td id="T_01116_row4_col1" class="data row4 col1">15.1435</td>
      <td id="T_01116_row4_col2" class="data row4 col2">3.8915</td>
      <td id="T_01116_row4_col3" class="data row4 col3">0.8222</td>
      <td id="T_01116_row4_col4" class="data row4 col4">0.2409</td>
      <td id="T_01116_row4_col5" class="data row4 col5">0.2069</td>
    </tr>
    <tr>
      <th id="T_01116_level0_row5" class="row_heading level0 row5">5</th>
      <td id="T_01116_row5_col0" class="data row5 col0">2.1601</td>
      <td id="T_01116_row5_col1" class="data row5 col1">14.5490</td>
      <td id="T_01116_row5_col2" class="data row5 col2">3.8143</td>
      <td id="T_01116_row5_col3" class="data row5 col3">0.8318</td>
      <td id="T_01116_row5_col4" class="data row5 col4">0.2372</td>
      <td id="T_01116_row5_col5" class="data row5 col5">0.2037</td>
    </tr>
    <tr>
      <th id="T_01116_level0_row6" class="row_heading level0 row6">6</th>
      <td id="T_01116_row6_col0" class="data row6 col0">2.1340</td>
      <td id="T_01116_row6_col1" class="data row6 col1">14.5847</td>
      <td id="T_01116_row6_col2" class="data row6 col2">3.8190</td>
      <td id="T_01116_row6_col3" class="data row6 col3">0.8269</td>
      <td id="T_01116_row6_col4" class="data row6 col4">0.2353</td>
      <td id="T_01116_row6_col5" class="data row6 col5">0.2010</td>
    </tr>
    <tr>
      <th id="T_01116_level0_row7" class="row_heading level0 row7">7</th>
      <td id="T_01116_row7_col0" class="data row7 col0">2.1856</td>
      <td id="T_01116_row7_col1" class="data row7 col1">13.6643</td>
      <td id="T_01116_row7_col2" class="data row7 col2">3.6965</td>
      <td id="T_01116_row7_col3" class="data row7 col3">0.8365</td>
      <td id="T_01116_row7_col4" class="data row7 col4">0.2379</td>
      <td id="T_01116_row7_col5" class="data row7 col5">0.2084</td>
    </tr>
    <tr>
      <th id="T_01116_level0_row8" class="row_heading level0 row8">8</th>
      <td id="T_01116_row8_col0" class="data row8 col0">2.1767</td>
      <td id="T_01116_row8_col1" class="data row8 col1">14.8072</td>
      <td id="T_01116_row8_col2" class="data row8 col2">3.8480</td>
      <td id="T_01116_row8_col3" class="data row8 col3">0.8144</td>
      <td id="T_01116_row8_col4" class="data row8 col4">0.2439</td>
      <td id="T_01116_row8_col5" class="data row8 col5">0.2122</td>
    </tr>
    <tr>
      <th id="T_01116_level0_row9" class="row_heading level0 row9">9</th>
      <td id="T_01116_row9_col0" class="data row9 col0">2.1280</td>
      <td id="T_01116_row9_col1" class="data row9 col1">13.4163</td>
      <td id="T_01116_row9_col2" class="data row9 col2">3.6628</td>
      <td id="T_01116_row9_col3" class="data row9 col3">0.8368</td>
      <td id="T_01116_row9_col4" class="data row9 col4">0.2345</td>
      <td id="T_01116_row9_col5" class="data row9 col5">0.2052</td>
    </tr>
    <tr>
      <th id="T_01116_level0_row10" class="row_heading level0 row10">Mean</th>
      <td id="T_01116_row10_col0" class="data row10 col0">2.1497</td>
      <td id="T_01116_row10_col1" class="data row10 col1">14.1745</td>
      <td id="T_01116_row10_col2" class="data row10 col2">3.7634</td>
      <td id="T_01116_row10_col3" class="data row10 col3">0.8289</td>
      <td id="T_01116_row10_col4" class="data row10 col4">0.2374</td>
      <td id="T_01116_row10_col5" class="data row10 col5">0.2062</td>
    </tr>
    <tr>
      <th id="T_01116_level0_row11" class="row_heading level0 row11">Std</th>
      <td id="T_01116_row11_col0" class="data row11 col0">0.0249</td>
      <td id="T_01116_row11_col1" class="data row11 col1">0.7920</td>
      <td id="T_01116_row11_col2" class="data row11 col2">0.1059</td>
      <td id="T_01116_row11_col3" class="data row11 col3">0.0083</td>
      <td id="T_01116_row11_col4" class="data row11 col4">0.0032</td>
      <td id="T_01116_row11_col5" class="data row11 col5">0.0032</td>
    </tr>
  </tbody>
</table>

</div>

</div>

</div>

</div>

</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-python"><pre><span></span><span class="n">plot_model</span><span class="p">(</span><span class="n">UberMLTunned</span><span class="p">)</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>




<div class="jp-RenderedImage jp-OutputArea-output ">
<img src="data:image/png;base64,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" class="
jp-needs-light-background
">
</div>

</div>

</div>

</div>

</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-python"><pre><span></span><span class="n">plot_model</span><span class="p">(</span><span class="n">UberMLTunned</span><span class="p">,</span> <span class="n">plot</span> <span class="o">=</span><span class="s1">'error'</span><span class="p">)</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>




<div class="jp-RenderedImage jp-OutputArea-output ">
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAREAAAEVCAYAAADHBdGYAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjUuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8qNh9FAAAACXBIWXMAAAsTAAALEwEAmpwYAACNVklEQVR4nOy9eZwcdZ3///xUVZ9zzySZyX2nEhLOcAkxhFNFDmURvBFEcRFxxWtd9ftz1V3XC3d1RXdRAREXDwQUPJCQgysc4UxIKskk5J57pqenj+o6Pr8/qrqne+77SOblAzPTXV1HT9Xr8z5fbyGlZApTmMIUhgplvE9gClOYwuTGFIlMYQpTGBamSGQKU5jCsDBFIlOYwhSGhSkSmcIUpjAsTJHIFKYwhWFBG+8TGCvour4AqAVez3tZAP9lGMYvhrnvR4DfG4Zxt67rrwDrDMNo62XbMuBBwzAu8H/vc/tBnsfXgE8Ch7u89VfDMP55uPvv47inAA8AMeAqwzDeHMI+1gH/bRjGql7enw/8K3AOYAEB4HfA1wzDsPzP/wUw/I9oQBPwj4Zh7PDf3wDcaxjGh7vsewNwhmEYxb3cJ8XAIeAGwzD2DvbajnUcNyTiI2UYxinZX3Rdnw1s03X9RcMwXhuJA+TvvxdUAGcOYvvB4jeGYdwywvvsD1cAGwzDuHE0du7/nbYA/w+43jAMqet6MXAP8H3gVn/T2i5/3y8APwQu9l86Clym63rUMIykv818QO9yyK73ifD382/A+0b26iY/jjcSKYBhGId1Xd8NLNN1/TTgo0AREDMM43xd1z8K3Izn9jUDtxiGsVPX9Vl4N/AsYD8wI7tPXdclMN0wjCZd178EXAfYwG7gI8BdQMS3QFb772W3/yreTWoDu/zj1em6vhF4FjgXmAc8CVxnGIY7mOv199MCLAd+AvxDl98f9P9dgGel3WMYxnf91flJYIf/3nmGYRz19/kB/ztSdV2PGIbxgX6uI3c8wzB+NMBT/2fgAcMw7sy+YBhGh67rtwBX93KtAqjEI44sWvCsjHcBv/Zf+7D/8yf6OH4YmAnU+/sOAt8GzgNU4GXgVsMw2nVdPxO4Awj6x5oP3Obv57+ABN49diZwCfAVf9sk8DnDMJ7VdX058HP/uAL4mWEYd/TxegC4HbgQcIDngM8YhhHXdf1N//eTgH8xDOPBPq5zSDiuYyK6rr8FWIL3JQOsxHMtztd1/Tw8AnirYRinAt8B/uBv92Ngi2EYK/FWweU97PsKPNJ4i2+i7wNuAa7HX+kMw3Dytr8eeAeeWX0SsA24O2+Xi4F1wInABXg3cE+4Vtf1V7r897a891sNwzgh7wHO//0+PIviRDzC+qCu6+/1t5sDfMMwjGVZAgEwDOM+4Kd4FtAHBnAdXY8/ELwV+FvXFw3DONplP4uz1wwcAf4R+M8uH/sl8KG836+lk1CyiPj7eU3X9XrgJTw36Yv++/+MR5CrDcM42T/Wf+i6ruG5dV/1r/2HwCl5+10FvM//zDzg34FL/fvr48AfdF0vAj4P/MkwjNXApcBaXdeVPl7/Ct6CdrL/nwJ8N++42wzDWDEaBALHnyWStQCg02f+gGEYB3VdB3jNMIx2//134hHMM/57AJW6rlcCFwGfAzAMY4+u60/0cKyLgN8ZhtHqb3cb5GIzPeEdwF2GYST83/8L+LK/6oF387hAXNf1PXirbE/oz515sqff/Zv3XLzVEcMwYrqu3+2f1xa8h+bZPvY70OvoevyBQAC5/gxd1z8PfMD/tQY4wf+5qzvzD8DfdF1flLevPwE/0XV9BrAU2IlnoeQj5874BPwr4DHDMDr89y8DyoGL/XsjCDTgETyGYfzF/3eDruvb8vZ70DCM/f7PF+NZN+vz7i8X7557EPilb9U8jmfluLqu9/b6O4AvG4Zh+ef8I+ChvOMO5TsfMI43EinwdXtAR97PKl4Q7osAPuPPAlrxbmiRt63dw75sCm/8crwbrzd0tQoVvL9P9jipvPe6Hn8w6Ojld6WHfSp4AUwA0zCMnq6zK/q7jq7HHwiewbPCHgEwDOO7+Cut7z72aFEbhvGArus/oZNkMAwjo+v6A3ju1koKraSe9vE3XddvB/5P1/UVhmHE8O6NT2fJwo/PZF2ert+hk/dz1/trvWEY12Zf0HV9LnDEMIxXdV1fikc0FwL/n67r5xiG8UhPr/dw/fl/t67HHXEc1+5MP3gMeJ+u6zP93z8BrPd//iue+Ymu6/OA83v4/OPAVbqul/q/fw3PN7bx4gddb7a/Adf7FgF4btJmwzDMEbiWfmEYRhzP4vgk5LJIHwb+PshdjcZ1/Btwja7rH9Z1XfXPT9V1/Rr//R5jQ7qun4tnJRhd3volnqu5Fu9v2R++B7ThZYfAu8ZbdF0P+ovLncC38GJGpq7rb/ePfyaeddJTl+sTwCV+nANd1y8FXgPCuq7/GrjWMIz78eJN7cDc3l73z+cTuq4H/PP5JIP/uw0Zx5slMmD4K9C3gb/ruu7i/cGu8jMDnwTu0nV9B17q75UePv9nXddPAJ72zdXtwMfwAmgvATv8mzyLn+PdEM/7N8IeOk32weBaXdfXdHntgGEYVwzgsx8AfuzHNYJ4MZK78YKDA8VwrmOFrutdV83ZhmEc0nX9bDwi/pyu6w4QxfsezzYMo8X/jhfnuasKYOL9zdryXAb84GUR8EfDMOz893qCn0K+Bc81+hnwDTxieRnPongF+Ky/r38Afqrr+rfwgsp1eH/zoi773K7r+seB+/0FxQauMAwjoev6N4Cf6bp+E54l8yCwCS+w29Prz/nn8wreM/088Kk+L2oEIaakAKYwhZGDruvfBb5nGEa97568CiwaiTqgiYopS2QKUxhZ7McLllp48ZEbj2UCgSlLZApTmMIwMRVYncIUpjAsTEp3ZuvWrSHgDLxqRKefzacwhSkMHipeyvqF1atX95lZG3MS8Ut078Ern3bwMhY2XhZA4lU4frKfku4zGOUCmilMYQqAVy38VF8bjIclcimgGYZxjq7rF+PVAASArxiGsVHX9Z8CV+Klr3rDUYBly5YRDAb72GziY9u2baxa1WPj6qTH1LVNPrz++uuceOKJZDIZdu3aBYW9Rz1iPEhkF6D5NQSleG3dZ+Plu8Fr576EvknEAQgGg4RCoVE81bHBsXANvWHq2iYPtm7dynPPPcfq1avzX+43XDAeJNKB58rsBKbh9SGsNQwjmyaKA2UD2dG2bdv632gSYOvWreN9CqOGqWubHNi5cyfPP/88AI8++igzZ87s5xOdGA8S+QzwN8MwvuQX4zyBVx2ZRQleiXG/WLVqVbfVwLZtXHdQHfLjiqz5OBmgKAqaNvBbZuvWrV1XtWMGx9K1bd26lYMHDzJz5kzmzp3LZZddhmmaA16kxyPF24qngAVe92QAeFn3lKfA6wIdUtA0Ho+TyWSGfYJjicWLF4/3KQwYmUyGeDw+3qcxhRHE1q1b+fvfvTabiy++mOXLu6la9IvxsER+APxC1/Un8SyQfwFeBO7028V3AL8f7E5t20ZVVaLR6Iie7GjDsqxJExwOBoMkk0ls2x6URTKFiYmuBLJ69eohuWhjfif4mgzX9PBWbyI7A4LrulM39hhAVdVJ5S5OoWc4jsPrr3syslkCGSqmnropDApCDFXGZAoTCaqqcu2117Jv3z5OOOGE/j/QB6bK3qcwheMI+/fvJ9svF4lEhk0gMEUiU5jCcYOtW7fyf//3f/ztb39jJBtvp0hkClM4DpAfRJ0+ffqIuqVTMZExxOOPP87GjRvp6Ojg6quvZs2argJkU5jCyKOnLMxIYsoSGQXcf//9nHvuuVxxxRVcdNFFPPTQQwBcdNFFfPOb3+Rf//Vf+fOf/zzk/W/evJm3ve1tXHzxxfzv//5vr9vdfffdvPOd7+Syyy7jtttuwzRNjh49yoc+9CEuvfRS3vnOd3LPPfcUfMZxHN71rndx0003Dfn8pjBxMNoEAlMkMirYtWsXt9xyC3/84x+5/fbb+da3vlXw/k9+8hM+8IGhyKd6D/nXv/51fvazn/Hoo4/yyCOPsGfPnm7b1dfX88tf/pIHHniARx55BMdxePTRR1FVlX/+53/mz3/+M7/5zW/49a9/XfD5X/7yl5OqAG4KveONN94YdQKBKRLBtB2OxJKY9sjJkhiGwcKFCwGYM2cOgYCn3i+l5Lvf/S5r165l5cqVQ9r3a6+9xvz585k7dy7BYJB3vvOdrF+/vsdtHcchnU5j2zbpdJoZM2YwY8aM3LGLi4tZtGgR9fX1ANTV1bFx40auvrrHoXJTmGRYtGgRM2fOHFUCgeM4JuK4Lj/cvIONtfU0J0yqikKsW1zNrWtXoCrD49Zdu3axcOFCpJT86le/4jOf+QwA9957L88++yzxeJz9+/fzvvcVjnV9//vfTyKR6La/L37xi5xzzjmAZ2HU1NTk3quurua117qPEa6uruaGG27g/PPPJxQKce6553aLwRw6dIgdO3Zw8sknA/Dv//7vfP7zn+/xHKYweSClRAhBOBzmgx/8IKqqjurxjlsS+eHmHfxx+yEUIQhpKh2mzR+3HwLgM+uGZiUAHD16lEQiwcc//nHq6+vRdZ1PfcpT7//whz/Mhz/84V4/++tfd53mOHTEYjHWr1/P+vXrKSkp4dOf/jQPP/wwV155JQCJRIJbb72Vf/mXf6G4uJgNGzZQWVnJqlWreO655/rZ+xQmKrZu3Up9fT3veMc7EEKMOoHAcUoipu2wsbYepUuaSxGCjbX13LxmOSFtaF/+rl27OP300/nlL39JLBbjsssu4+WXX+a0007r97MDsUSqq6upq6vLvVdfX091dXW3zzzzzDPMmTOHykpv2uYll1zCyy+/zJVXXollWdx6661cfvnlXHLJJQC89NJLPPHEE2zevBnTNOno6OBzn/sc3/ve94b0PUxh7JEfRD3hhBNYsGDBmBz3uCSR5oRJc8LskShakt57s8qG1shnGEauCrCsrIzLLruMTZs2DYhEBmKJnHjiibz55pscPHiQ6upqHn30Ub7//e93227WrFm8+uqrpFIpwuEwzz77LKtWrUJKyZe//GUWLVrE9ddfn9v+s5/9LJ/97GcBeO655/jFL34xRSCTCF2zMGNFIHCcBlarikJUFfWsSlUZ7f29gcAwDFasWJH7/YILLmDTpk19fGJw0DSN//f//h833ngjl156Ke94xztYunRp7v2Pfexj1NfXc/LJJ/O2t72Nd7/73Vx++eW4rsu1117L1q1befjhh9myZQtXXnklV1555Yie3xTGHmORxu0Lk3LuzNatWxcA+/JFibI6IgNtq//Bxu25mEgWrpRcsXLOsGIig0UikaCoqKj/DScIBvM9H0vCPV0xUa5tpAkke115okQLV69e/WZfnzku3RmAW9d61sLG2npakiaV0c7szBSmMBngOA7bt28HxscCyeK4JRFVUfjMupXcvGZ5LsU71GDqFKYwHlBVlWuuuYa9e/eOSDfuUHFcxkTyEdJUZpVFpwhkCpMGb775Zk4YKhwOjyuBwBSJTGEKkwpbt27l/vvv5y9/+cuItvMPB1MkMoUpTBLkB1FramomjMrcFIlMYQqTAOOdxu0L4zGL9yPAR/xfw8ApwDrgv/Bm8j5mGMa/jvV5TWEKExUTmUBgHCwRwzDuNgxjnWEY64CtwK3AT4H3A2uAs3RdP3Wsz2sk8Ic//GHYVZ6/+tWvur128OBB3v72t/PFL36Rf/u3f+PIkSO0tbXxpz/9aVjHmsLEx44dOyY0gcA4pnh1XT8dWAl8CfiMYRi1/ut/Ay4CXu5vH10ndC1evBjLskb+ZAcI0zSxLGvQXbD5299xxx28+93vLnj/mWee4dxzz+W2227Lvfbiiy/y2GOPccEFFwzvpAcJy7Kora0d8PbH0qjJrhiLa8tkMliWxaJFi8bsmIM9xnjWifwL8K94Q73b816PA4sGsoP+KlazzWc94fbbb+cjH/kI4CmA5T+gXdHS0jKQ0yEUCrF9+3ZuvvlmOjo6+NSnPsW6det4/vnn+cEPfoCqqsydO5evf/3rHDp0iC996UsIIRBC8P3vf5+HHnqI9vZ2vvvd7/K1r30NgCNHjnDXXXeRTqdZvHgxf/nLX/ja177G3Xffzc6dO3nkkUe49tprB3R+I4FMJsOJJ544VbE6yteWbecHOOuss1CGKU8xUPRQsdovxiWwqut6OaAbhrEBj0BK8t4e8CzeiYhIJMLdd9/N//7v//L1r38dx3H46le/yn//93/zq1/9iurqah588EGeeeYZTjrpJH7yk5/wqU99ing8zj/+4z9SVlaWIxDwGuk+/vGPc9lll/H+978/9/onPvEJzj777DElkCmMDbZu3cqjjz6aqwUZKwIZKsbLElkLrAcwDKNd1/WMruuLgb3A2/AslGFjoBbERz7ykZxVMlysXr0aIQRVVVWUlJTQ2tpKQ0MD//RP/wRAOp3mnHPO4eabb+bOO+/klltuoby8PCdcNIXjG/lB1JUrV+YU8iYyxotEdDzCyOITwH2AipedmbSqONnRhI2NjSSTSSoqKqipqeGOO+6gpKSE9evXE41GWb9+PatXr+b6669nw4YN/OxnP+Nb3/rWgAuIFEWZGmd5jKFrFmYyEAiME4kYhvHdLr9vAc4ej3MZaaTTaT784Q+TTCb5+te/jqqqfPnLX+bjH/84UkqKior4zne+QyKR4Itf/CKKoiCE4Etf+hLgBYcHIgY0b948du3axd133z1iVtQUxg8TKY3rOA4/+tGPOP/88we0/XErBTBRMCUFMDkxktc2kQhk8+bN3HHHHbz66qvZcoMpKYApTGEiw3Vd3njjDWD8CWT37t186lOf4uDBgyxbtmzAn5sikSlMYRyhKEqunT9fEW888PLLL3Pw4EFOOOEE7rnnHtra2gb0uSkSmcIUxgH79u1j/vz5KIpCKBQadwIBcmR2yy23EAgEBkwiEzsBPYUpHIPYunUrv/nNb3j00UfHtZ0/mUzyqU99qqCo7OKLL6a4uHhQ+5myRKYwhTFEfhB11qxZ49bOf+jQIT70oQ/x6quv8tJLL/Hkk08OuahtyhKZwhTGCBMlC7NlyxYuvPBCXn31VRYsWMCdd945rKrYKRIZQWzevJnf/OY3Ba9dc801HDp0aFD7MU2T3/3ud4DXGZydtdtTh+8UJgcmCoHcfffdXHnllTQ2NnLeeeexfv36YcsrTpHICGLt2rUj0svS2NiYI5GrrrqKCy+8EICf/OQnw973FMYeO3funBAE8tWvfpXbbrsNy7L4x3/8R373u99RUVEx7P0e0zGR//iP/+j1vbe//e2ccsopALzyyiv89a9/7XXbf/7nfx7Q8f7whz+wd+9eVFXlySefpKamhtbWVgDi8Thf/vKXc79/5StfQdd1rrzySk4//XT27dtHVVUVP/rRj/jpT3/Knj17+O///m+klEybNo22tjZisRhf+9rXiMfjXH755axbt47a2lq+/e1v87//+78D/FamMNZYuHAhc+bMYcWKFeNaB3LqqacSCoW4/fbbuw2THw6mLJERxrZt23jhhRf4/e9/nytvB/jpT3/K2Wefzb333ss3vvGNXKfu4cOH+fSnP81vfvMbWlpaeP311/nEJz7BkiVLuOWWW3L7ze/wfc973sODDz4IwO9//3uuvvrqMb/OKfSPbOYlFArx/ve/f1wIJB6P536+6qqr2Lp164gSCBzjlshALYhTTjklZ5UMF0ePHuW8885DURSKi4tzlX+7du1iy5Yt/OUvfwEgFosBUF5ezsyZMwGYOXMmpmn2e4yzzjqLb37zm7S0tPD000/3qYUyhfHB1q1bOXLkCO985ztRFGVc2vkffPBBPve5z/H73/+eU0/1xAJnzZo14sc5pklkPDBnzhxee+01XNclnU6zZ88eABYtWsQVV1zB5ZdfTnNzcy7m0VOKr7cO3ezKJoTgiiuu4Jvf/CbnnnsugUBgFK9oCoNF13b+rCrZWMFxHP793/+dH/zgBwA8/PDDORIZDUyRyAhjxYoVRKNRrr76ambMmEFVVRXgiQh9+ctf5re//S0dHR0FrkpXVFVVYVkW3/3udwmHw7nX8zt8r7rqKtatW8fDDz886tc0hYGjaxZmrAmkvb2dj3/84zz22GOoqso3vvENbrrpplE95hSJjCCuuuqq3M8333xzt/fvuOOObq9lbzggt3IAPZLDvffem/vZcRxWr17N4sWLh3y+UxhZjHcad8+ePXzgAx9g9+7dlJeX84tf/IJ169aN+nGnAquTEI899hg33ngjt95663ifyhR8jDeBmKbJu9/9bnbv3s3y5ctZv379mBAITFkikxKXXHIJl1xyybgcO19AeAoe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" class="
jp-needs-light-background
">
</div>

</div>

</div>

</div>

</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-python"><pre><span></span><span class="n">plot_model</span><span class="p">(</span><span class="n">UberMLTunned</span><span class="p">,</span> <span class="n">plot</span> <span class="o">=</span><span class="s1">'feature_all'</span><span class="p">)</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>




<div class="jp-RenderedImage jp-OutputArea-output ">
<img src="data:image/png;base64,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" class="
jp-needs-light-background
">
</div>

</div>

</div>

</div>

</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-python"><pre><span></span><span class="n">plot_model</span><span class="p">(</span><span class="n">UberMLTunned</span><span class="p">,</span> <span class="n">plot</span> <span class="o">=</span><span class="s1">'learning'</span><span class="p">)</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>




<div class="jp-RenderedImage jp-OutputArea-output ">
<img src="data:image/png;base64,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" class="
jp-needs-light-background
">
</div>

</div>

</div>

</div>

</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs  ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-python"><pre><span></span><span class="n">UberMLFinal</span> <span class="o">=</span> <span class="n">finalize_model</span><span class="p">(</span><span class="n">UberMLTunned</span><span class="p">)</span>
</pre></div>

     </div>
</div>
</div>
</div>

</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs  ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-python"><pre><span></span><span class="n">jsCode</span> <span class="o">=</span> <span class="n">convert_model</span><span class="p">(</span><span class="n">UberMLTunned</span><span class="p">,</span> <span class="s1">'javascript'</span><span class="p">)</span>
<span class="n">arquivoJs</span> <span class="o">=</span> <span class="nb">open</span><span class="p">(</span><span class="s1">'scoreModel.js'</span><span class="p">,</span> <span class="s1">'a+'</span><span class="p">)</span>
<span class="n">arquivoJs</span><span class="o">.</span><span class="n">write</span><span class="p">(</span><span class="s1">'export '</span><span class="o">+</span><span class="n">jsCode</span><span class="p">)</span>
<span class="n">arquivoJs</span><span class="o">.</span><span class="n">close</span><span class="p">()</span>
</pre></div>

     </div>
</div>
</div>
</div>

</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-python"><pre><span></span><span class="n">save_model</span><span class="p">(</span><span class="n">UberMLFinal</span><span class="p">,</span> <span class="s1">'UberPrice'</span><span class="p">)</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>


<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain">
<pre>Transformation Pipeline and Model Successfully Saved
</pre>
</div>
</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt">Out[&nbsp;]:</div>




<div class="jp-RenderedText jp-OutputArea-output jp-OutputArea-executeResult" data-mime-type="text/plain">
<pre>(Pipeline(memory=None,
          steps=[('dtypes',
                  DataTypes_Auto_infer(categorical_features=[],
                                       display_types=True, features_todrop=[],
                                       id_columns=[], ml_usecase='regression',
                                       numerical_features=[],
                                       target='fare_amount', time_features=[])),
                 ('imputer',
                  Simple_Imputer(categorical_strategy='not_available',
                                 fill_value_categorical=None,
                                 fill_value_numerical=None,
                                 numeric_stra...
                                boosting_type='gbdt', class_weight=None,
                                colsample_bytree=1.0, feature_fraction=0.4,
                                importance_type='split', learning_rate=0.3,
                                max_depth=-1, min_child_samples=1,
                                min_child_weight=0.001, min_split_gain=0.9,
                                n_estimators=60, n_jobs=-1, num_leaves=4,
                                objective=None, random_state=4673, reg_alpha=1,
                                reg_lambda=1e-07, silent='warn', subsample=1.0,
                                subsample_for_bin=200000, subsample_freq=0)]],
          verbose=False),
 'UberPrice.pkl')</pre>
</div>

</div>

</div>

</div>

</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs  ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-python"><pre><span></span><span class="n">dfUberSam</span> <span class="o">=</span> <span class="n">dfUber</span><span class="o">.</span><span class="n">sample</span><span class="p">(</span><span class="n">frac</span><span class="o">=</span><span class="mf">0.001</span><span class="p">)</span>
</pre></div>

     </div>
</div>
</div>
</div>

</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-python"><pre><span></span><span class="n">dtLoad</span> <span class="o">=</span> <span class="n">load_model</span><span class="p">(</span><span class="s1">'UberPrice'</span><span class="p">)</span>
<span class="n">predict_model</span><span class="p">(</span><span class="n">dtLoad</span><span class="p">,</span> <span class="n">data</span><span class="o">=</span><span class="n">dfUberSam</span><span class="p">)</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>


<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain">
<pre>Transformation Pipeline and Model Successfully Loaded
</pre>
</div>
</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>



<div class="jp-RenderedHTMLCommon jp-RenderedHTML jp-OutputArea-output " data-mime-type="text/html">
<style type="text/css">
</style>
<table id="T_42e26">
  <thead>
    <tr>
      <th class="blank level0">&nbsp;</th>
      <th id="T_42e26_level0_col0" class="col_heading level0 col0">Model</th>
      <th id="T_42e26_level0_col1" class="col_heading level0 col1">MAE</th>
      <th id="T_42e26_level0_col2" class="col_heading level0 col2">MSE</th>
      <th id="T_42e26_level0_col3" class="col_heading level0 col3">RMSE</th>
      <th id="T_42e26_level0_col4" class="col_heading level0 col4">R2</th>
      <th id="T_42e26_level0_col5" class="col_heading level0 col5">RMSLE</th>
      <th id="T_42e26_level0_col6" class="col_heading level0 col6">MAPE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_42e26_level0_row0" class="row_heading level0 row0">0</th>
      <td id="T_42e26_row0_col0" class="data row0 col0">Light Gradient Boosting Machine</td>
      <td id="T_42e26_row0_col1" class="data row0 col1">2.5974</td>
      <td id="T_42e26_row0_col2" class="data row0 col2">28.9494</td>
      <td id="T_42e26_row0_col3" class="data row0 col3">5.3805</td>
      <td id="T_42e26_row0_col4" class="data row0 col4">0.7291</td>
      <td id="T_42e26_row0_col5" class="data row0 col5">0.2972</td>
      <td id="T_42e26_row0_col6" class="data row0 col6">0.2282</td>
    </tr>
  </tbody>
</table>

</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt">Out[&nbsp;]:</div>



<div class="jp-RenderedHTMLCommon jp-RenderedHTML jp-OutputArea-output jp-OutputArea-executeResult" data-mime-type="text/html">
<div>
<style scoped="">
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fare_amount</th>
      <th>passenger_count</th>
      <th>Distance</th>
      <th>Hour</th>
      <th>Minute</th>
      <th>Day</th>
      <th>Label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>63189</th>
      <td>12.9</td>
      <td>1</td>
      <td>2.577767</td>
      <td>22</td>
      <td>57</td>
      <td>6</td>
      <td>8.774656</td>
    </tr>
    <tr>
      <th>134409</th>
      <td>7.0</td>
      <td>1</td>
      <td>2.849883</td>
      <td>13</td>
      <td>44</td>
      <td>1</td>
      <td>10.236725</td>
    </tr>
    <tr>
      <th>30130</th>
      <td>5.5</td>
      <td>1</td>
      <td>0.715193</td>
      <td>14</td>
      <td>28</td>
      <td>1</td>
      <td>5.829010</td>
    </tr>
    <tr>
      <th>45940</th>
      <td>6.0</td>
      <td>1</td>
      <td>1.050148</td>
      <td>1</td>
      <td>44</td>
      <td>5</td>
      <td>4.903515</td>
    </tr>
    <tr>
      <th>60229</th>
      <td>11.4</td>
      <td>1</td>
      <td>4.947954</td>
      <td>2</td>
      <td>14</td>
      <td>1</td>
      <td>13.297933</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>164383</th>
      <td>6.9</td>
      <td>1</td>
      <td>1.406125</td>
      <td>14</td>
      <td>42</td>
      <td>5</td>
      <td>6.782369</td>
    </tr>
    <tr>
      <th>168893</th>
      <td>7.3</td>
      <td>1</td>
      <td>1.957648</td>
      <td>14</td>
      <td>6</td>
      <td>6</td>
      <td>8.758213</td>
    </tr>
    <tr>
      <th>27652</th>
      <td>4.5</td>
      <td>1</td>
      <td>0.935871</td>
      <td>22</td>
      <td>51</td>
      <td>3</td>
      <td>5.138636</td>
    </tr>
    <tr>
      <th>100084</th>
      <td>6.5</td>
      <td>1</td>
      <td>0.807171</td>
      <td>9</td>
      <td>4</td>
      <td>1</td>
      <td>5.540899</td>
    </tr>
    <tr>
      <th>105139</th>
      <td>9.0</td>
      <td>2</td>
      <td>1.214571</td>
      <td>16</td>
      <td>18</td>
      <td>1</td>
      <td>7.089686</td>
    </tr>
  </tbody>
</table>
<p>174 rows × 7 columns</p>
</div>
</div>

</div>

</div>

</div>

</div>









</body></html>