class Screen {
	el = null
	constructor(id) {
		this.el = document.getElementById(id)
	}

	show() {
		this.el.classList.remove('screen--hide')
	}
	hide() {
		this.el.classList.add('screen--hide')
	}
}
export { Screen }
