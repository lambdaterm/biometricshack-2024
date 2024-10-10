"use strict";

function __$styleInject(css) {
    if (css && "undefined" != typeof window) {
        var style = document.createElement("style");
        style.setAttribute("media", "screen");
        style.innerHTML = css;
        document.head.appendChild(style);
    }
}

function escapeString(e) {
    return e.replace(/"/g, '\\"')
}

function getType(e) {
    return null === e ? "null" : typeof e
}

function isObject(e) {
    return !!e && "object" == typeof e
}

function getObjectName(e) {
    if (void 0 === e) return "";
    if (null === e) return "Object";
    if ("object" == typeof e && !e.constructor) return "Object";
    var t = /function ([^(]*)/.exec(e.constructor.toString());
    return t && t.length > 1 ? t[1] : ""
}

function getValuePreview(e, t, r) {
    return "null" === e || "undefined" === e ? e : ("string" !== e && "stringifiable" !== e || (r = '"' + escapeString(r) + '"'), "function" === e ? t.toString().replace(/[\r\n]/g, "").replace(/\{.*\}/, "") + "{…}" : r)
}

function getPreview(e) {
    var t = "";
    return isObject(e) ? (t = getObjectName(e), Array.isArray(e) && (t += "[" + e.length + "]")) : t = getValuePreview(getType(e), e, e), t
}

function cssClass(e) {
    return "json-formatter-" + e
}

function createElement(e, t, r) {
    var n = document.createElement(e);
    return t && n.classList.add(cssClass(t)), void 0 !== r && (r instanceof Node ? n.appendChild(r) : n.appendChild(document.createTextNode(String(r)))), n
}

__$styleInject(`
.json-formatter-row {
 font-family: Inter;
}

.json-formatter-row,
.json-formatter-row a,
.json-formatter-row a:hover {
 color: black;
 text-decoration: none;
}

.json-formatter-row .json-formatter-row {
 margin: 0 1rem;
}

.json-formatter-row .json-formatter-children.json-formatter-empty {
 opacity: 0.5;
 margin-left: 1rem;
}

.json-formatter-row .json-formatter-children.json-formatter-empty:after {
 display: none;
}

.json-formatter-row .json-formatter-children.json-formatter-empty.json-formatter-object:after {
 content: "No properties";
}

.json-formatter-row .json-formatter-children.json-formatter-empty.json-formatter-array:after {
 content: "[]";
}

.json-formatter-row .json-formatter-string,
.json-formatter-row .json-formatter-stringifiable {
 color: green;
 word-break: break-word;
}

.json-formatter-row .json-formatter-number {
 color: blue;
}

.json-formatter-row .json-formatter-boolean {
 color: red;
}

.json-formatter-row .json-formatter-null {
 color: #855A00;
}

.json-formatter-row .json-formatter-undefined {
 color: #ca0b69;
}

.json-formatter-row .json-formatter-function {
 color: #FF20ED;
}

.json-formatter-row .json-formatter-date {
 background-color: rgba(0, 0, 0, 0.05);
}

.json-formatter-row .json-formatter-url {
 text-decoration: underline;
 color: blue;
 cursor: pointer;
}

.json-formatter-row .json-formatter-bracket {
 color: blue;
}

.json-formatter-row .json-formatter-key {
 color: #00008B;
 padding-right: 0.2rem;
}

.json-formatter-row .json-formatter-toggler-link {
 cursor: pointer;
}

.json-formatter-row .json-formatter-toggler {
 line-height: 1.2rem;
 font-size: 0.7rem;
 vertical-align: middle;
 opacity: 0.6;
 cursor: pointer;
 padding-right: 0.2rem;
}

.json-formatter-row .json-formatter-toggler:after {
 display: inline-block;
 transition: transform 100ms ease-in;
 content: "►";
}

.json-formatter-row > a > .json-formatter-preview-text {
 opacity: 0;
 transition: opacity 0.15s ease-in;
 font-style: italic;
}

.json-formatter-row:hover > a > .json-formatter-preview-text {
 opacity: 0.6;
}

.json-formatter-row.json-formatter-open > .json-formatter-toggler-link .json-formatter-toggler:after {
 transform: rotate(90deg);
}

.json-formatter-row.json-formatter-open > .json-formatter-children:after {
 display: inline-block;
}

.json-formatter-row.json-formatter-open > a > .json-formatter-preview-text {
 display: none;
}

.json-formatter-row.json-formatter-open.json-formatter-empty:after {
 display: block;
}

.json-formatter-dark.json-formatter-row {
 font-family: Inter;
}

.json-formatter-dark.json-formatter-row,
.json-formatter-dark.json-formatter-row a,
.json-formatter-dark.json-formatter-row a:hover {
 color: white;
 text-decoration: none;
}

.json-formatter-dark.json-formatter-row .json-formatter-row {
 margin-left: 1rem;
}

.json-formatter-dark.json-formatter-row .json-formatter-children.json-formatter-empty {
 opacity: 0.5;
 margin-left: 1rem;
}

.json-formatter-dark.json-formatter-row .json-formatter-children.json-formatter-empty:after {
 display: none;
}

.json-formatter-dark.json-formatter-row .json-formatter-children.json-formatter-empty.json-formatter-object:after {
 content: "No properties";
}

.json-formatter-dark.json-formatter-row .json-formatter-children.json-formatter-empty.json-formatter-array:after {
 content: "[]";
}

.json-formatter-dark.json-formatter-row .json-formatter-string,
.json-formatter-dark.json-formatter-row .json-formatter-stringifiable {
 color: #31F031;
 word-break: break-word;
}

.json-formatter-dark.json-formatter-row .json-formatter-number {
 color: #66C2FF;
}

.json-formatter-dark.json-formatter-row .json-formatter-boolean {
 color: #EC4242;
}

.json-formatter-dark.json-formatter-row .json-formatter-null {
 color: #EEC97D;
}

.json-formatter-dark.json-formatter-row .json-formatter-undefined {
 color: #ef8fbe;
}

.json-formatter-dark.json-formatter-row .json-formatter-function {
 color: #FD48CB;
}

.json-formatter-dark.json-formatter-row .json-formatter-date {
 background-color: rgba(255, 255, 255, 0.05);
}

.json-formatter-dark.json-formatter-row .json-formatter-url {
 text-decoration: underline;
 color: #027BFF;
 cursor: pointer;
}

.json-formatter-dark.json-formatter-row .json-formatter-bracket {
 color: #9494FF;
}

.json-formatter-dark.json-formatter-row .json-formatter-key {
 color: #23A0DB;
 padding-right: 0.2rem;
}

.json-formatter-dark.json-formatter-row .json-formatter-toggler-link {
 cursor: pointer;
}

.json-formatter-dark.json-formatter-row .json-formatter-toggler {
 line-height: 1.2rem;
 font-size: 0.7rem;
 vertical-align: middle;
 opacity: 0.6;
 cursor: pointer;
 padding-right: 0.2rem;
}

.json-formatter-dark.json-formatter-row .json-formatter-toggler:after {
 display: inline-block;
 transition: transform 100ms ease-in;
 content: "►";
}

.json-formatter-dark.json-formatter-row > a > .json-formatter-preview-text {
 opacity: 0;
 transition: opacity 0.15s ease-in;
 font-style: italic;
}

.json-formatter-dark.json-formatter-row:hover > a > .json-formatter-preview-text {
 opacity: 0.6;
}

.json-formatter-dark.json-formatter-row.json-formatter-open > .json-formatter-toggler-link .json-formatter-toggler:after {
 transform: rotate(90deg);
}

.json-formatter-dark.json-formatter-row.json-formatter-open > .json-formatter-children:after {
 display: inline-block;
}

.json-formatter-dark.json-formatter-row.json-formatter-open > a > .json-formatter-preview-text {
 display: none;
}

.json-formatter-dark.json-formatter-row.json-formatter-open.json-formatter-empty:after {
 display: block;
}
`);
var DATE_STRING_REGEX = /(^\d{1,4}[\.|\\/|-]\d{1,2}[\.|\\/|-]\d{1,4})(\s*(?:0?[1-9]:[0-5]|1(?=[012])\d:[0-5])\d\s*[ap]m)?$/,
    PARTIAL_DATE_REGEX = /\d{2}:\d{2}:\d{2} GMT-\d{4}/, JSON_DATE_REGEX = /\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}.\d{3}Z/,
    MAX_ANIMATED_TOGGLE_ITEMS = 10, requestAnimationFrame = window.requestAnimationFrame || function (e) {
        return e(), 0
    }, _defaultConfig = {
        hoverPreviewEnabled: !1,
        hoverPreviewArrayCount: 100,
        hoverPreviewFieldCount: 5,
        animateOpen: !0,
        animateClose: !0,
        theme: null,
        useToJSON: !0,
        sortPropertiesBy: null
    }, JSONFormatter = function () {
        function e(e, t, r, n) {
            void 0 === t && (t = 1), void 0 === r && (r = _defaultConfig), this.json = e, this.open = t, this.config = r, this.key = n, this._isOpen = null, void 0 === this.config.hoverPreviewEnabled && (this.config.hoverPreviewEnabled = _defaultConfig.hoverPreviewEnabled), void 0 === this.config.hoverPreviewArrayCount && (this.config.hoverPreviewArrayCount = _defaultConfig.hoverPreviewArrayCount), void 0 === this.config.hoverPreviewFieldCount && (this.config.hoverPreviewFieldCount = _defaultConfig.hoverPreviewFieldCount), void 0 === this.config.useToJSON && (this.config.useToJSON = _defaultConfig.useToJSON), "" === this.key && (this.key = '""')
        }

        return Object.defineProperty(e.prototype, "isOpen", {
            get: function () {
                return null !== this._isOpen ? this._isOpen : this.open > 0
            }, set: function (e) {
                this._isOpen = e
            }, enumerable: !0, configurable: !0
        }), Object.defineProperty(e.prototype, "isDate", {
            get: function () {
                return this.json instanceof Date || "string" === this.type && (DATE_STRING_REGEX.test(this.json) || JSON_DATE_REGEX.test(this.json) || PARTIAL_DATE_REGEX.test(this.json))
            }, enumerable: !0, configurable: !0
        }), Object.defineProperty(e.prototype, "isUrl", {
            get: function () {
                return "string" === this.type && 0 === this.json.indexOf("http")
            }, enumerable: !0, configurable: !0
        }), Object.defineProperty(e.prototype, "isArray", {
            get: function () {
                return Array.isArray(this.json)
            }, enumerable: !0, configurable: !0
        }), Object.defineProperty(e.prototype, "isObject", {
            get: function () {
                return isObject(this.json)
            }, enumerable: !0, configurable: !0
        }), Object.defineProperty(e.prototype, "isEmptyObject", {
            get: function () {
                return !this.keys.length && !this.isArray
            }, enumerable: !0, configurable: !0
        }), Object.defineProperty(e.prototype, "isEmpty", {
            get: function () {
                return this.isEmptyObject || this.keys && !this.keys.length && this.isArray
            }, enumerable: !0, configurable: !0
        }), Object.defineProperty(e.prototype, "useToJSON", {
            get: function () {
                return this.config.useToJSON && "stringifiable" === this.type
            }, enumerable: !0, configurable: !0
        }), Object.defineProperty(e.prototype, "hasKey", {
            get: function () {
                return void 0 !== this.key
            }, enumerable: !0, configurable: !0
        }), Object.defineProperty(e.prototype, "constructorName", {
            get: function () {
                return getObjectName(this.json)
            }, enumerable: !0, configurable: !0
        }), Object.defineProperty(e.prototype, "type", {
            get: function () {
                return this.config.useToJSON && this.json && this.json.toJSON ? "stringifiable" : getType(this.json)
            }, enumerable: !0, configurable: !0
        }), Object.defineProperty(e.prototype, "keys", {
            get: function () {
                if (this.isObject) {
                    var e = Object.keys(this.json);
                    return !this.isArray && this.config.sortPropertiesBy ? e.sort(this.config.sortPropertiesBy) : e
                }
                return []
            }, enumerable: !0, configurable: !0
        }), e.prototype.toggleOpen = function () {
            this.isOpen = !this.isOpen, this.element && (this.isOpen ? this.appendChildren(this.config.animateOpen) : this.removeChildren(this.config.animateClose), this.element.classList.toggle(cssClass("open")))
        }, e.prototype.openAtDepth = function (e) {
            void 0 === e && (e = 1), e < 0 || (this.open = e, this.isOpen = 0 !== e, this.element && (this.removeChildren(!1), 0 === e ? this.element.classList.remove(cssClass("open")) : (this.appendChildren(this.config.animateOpen), this.element.classList.add(cssClass("open")))))
        }, e.prototype.getInlinepreview = function () {
            var e = this;
            if (this.isArray) return this.json.length > this.config.hoverPreviewArrayCount ? "Array[" + this.json.length + "]" : "[" + this.json.map(getPreview).join(", ") + "]";
            var t = this.keys, r = t.slice(0, this.config.hoverPreviewFieldCount).map((function (t) {
                return t + ":" + getPreview(e.json[t])
            })), n = t.length >= this.config.hoverPreviewFieldCount ? "…" : "";
            return "{" + r.join(", ") + n + "}"
        }, e.prototype.render = function () {
            this.element = createElement("div", "row");
            var e = this.isObject ? createElement("a", "toggler-link") : createElement("span");
            if (this.isObject && !this.useToJSON && e.appendChild(createElement("span", "toggler")), this.hasKey && e.appendChild(createElement("span", "key", this.key + ":")), this.isObject && !this.useToJSON) {
                var t = createElement("span", "value"), r = createElement("span"),
                    n = createElement("span", "constructor-name", this.constructorName);
                if (r.appendChild(n), this.isArray) {
                    var o = createElement("span");
                    o.appendChild(createElement("span", "bracket", "[")), o.appendChild(createElement("span", "number", this.json.length)), o.appendChild(createElement("span", "bracket", "]")), r.appendChild(o)
                }
                t.appendChild(r), e.appendChild(t)
            } else {
                (t = this.isUrl ? createElement("a") : createElement("span")).classList.add(cssClass(this.type)), this.isDate && t.classList.add(cssClass("date")), this.isUrl && (t.classList.add(cssClass("url")), t.setAttribute("href", this.json));
                var s = getValuePreview(this.type, this.json, this.useToJSON ? this.json.toJSON() : this.json);
                t.appendChild(document.createTextNode(s)), e.appendChild(t)
            }
            if (this.isObject && this.config.hoverPreviewEnabled) {
                var i = createElement("span", "preview-text");
                i.appendChild(document.createTextNode(this.getInlinepreview())), e.appendChild(i)
            }
            var a = createElement("div", "children");
            return this.isObject && a.classList.add(cssClass("object")), this.isArray && a.classList.add(cssClass("array")), this.isEmpty && a.classList.add(cssClass("empty")), this.config && this.config.theme && this.element.classList.add(cssClass(this.config.theme)), this.isOpen && this.element.classList.add(cssClass("open")), this.element.appendChild(e), this.element.appendChild(a), this.isObject && this.isOpen && this.appendChildren(), this.isObject && !this.useToJSON && e.addEventListener("click", this.toggleOpen.bind(this)), this.element
        }, e.prototype.appendChildren = function (t) {
            var r = this;
            void 0 === t && (t = !1);
            var n = this.element.querySelector("div." + cssClass("children"));
            if (n && !this.isEmpty) if (t) {
                var o = 0, s = function () {
                    var t = r.keys[o], i = new e(r.json[t], r.open - 1, r.config, t);
                    n.appendChild(i.render()), (o += 1) < r.keys.length && (o > MAX_ANIMATED_TOGGLE_ITEMS ? s() : requestAnimationFrame(s))
                };
                requestAnimationFrame(s)
            } else this.keys.forEach((function (t) {
                var o = new e(r.json[t], r.open - 1, r.config, t);
                n.appendChild(o.render())
            }))
        }, e.prototype.removeChildren = function (e) {
            void 0 === e && (e = !1);
            var t = this.element.querySelector("div." + cssClass("children"));
            if (e) {
                var r = 0, n = function () {
                    t && t.children.length && (t.removeChild(t.children[0]), (r += 1) > MAX_ANIMATED_TOGGLE_ITEMS ? n() : requestAnimationFrame(n))
                };
                requestAnimationFrame(n)
            } else t && (t.innerHTML = "")
        }, e
    }();
export default JSONFormatter;