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
	showLoading() {
		this.el.classList.add('screen--loading')
	}
	hideLoading() {
		this.el.classList.remove('screen--loading')
	}
	toggleLoading() {
		this.el.classList.toggle('screen--loading')
	}
}
export { Screen }
