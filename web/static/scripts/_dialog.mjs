export class Dialog {
	dialogParams = {
		modalOverlay: document.getElementById('modal-overlay'),
		loadingModal: document.getElementById('loading-modal'),
		errorModal: document.getElementById('error-modal'),
		loadingCancelButton: document.getElementById('loading-cancel-button'),
		closeErrorModalButton: document.getElementById('close-error-modal-button'),
		closeAddNewFileModalButton: document.getElementById('close-add-new-file-modal-button'),
		errorConfirmButton: document.getElementById('error-confirm-button'),
		loadingMessage: document.querySelector('.loading-message'),
		errorMessage: document.querySelector('.error-message'),
		addNewFileButton: document.querySelector('.add-new-file-button'),
		addNewFileModal: document.querySelector('.add-new-file-modal')
	}
	constructor() {
		this.modalOverlay = this.dialogParams.modalOverlay
		this.loadingModal = this.dialogParams.loadingModal
		this.errorModal = this.dialogParams.errorModal
		this.loadingCancelButton = this.dialogParams.loadingCancelButton
		this.closeErrorModalButton = this.dialogParams.closeErrorModalButton
		this.errorConfirmButton = this.dialogParams.errorConfirmButton
		this.loadingMessage = this.dialogParams.loadingMessage
		this.errorMessage = this.dialogParams.errorMessage
		this.addNewFileButton = this.dialogParams.addNewFileButton
		this.closeAddNewFileModalButton = this.dialogParams.closeAddNewFileModalButton
		this.addNewFileModal = this.dialogParams.addNewFileModal

		this.loadingCancelButton.addEventListener('click', this.hideModalOverlay.bind(this))
		this.closeErrorModalButton.addEventListener('click', this.hideModalOverlay.bind(this))
		this.errorConfirmButton.addEventListener('click', this.hideModalOverlay.bind(this))
		this.addNewFileButton.addEventListener('click', this.showAddNewFileModal.bind(this))
		this.closeAddNewFileModalButton.addEventListener('click', this.hideModalOverlay.bind(this))
	}

	isShowModal(isLoadingModal) {
		if (isLoadingModal) {
			this.showModalOverlay()
			this.showLoadingModal()
		}
		if (!isLoadingModal) {
			this.hideModalOverlay()
		}
	}

	showModalOverlay() {
		this.modalOverlay.classList.add('modal--show')
	}

	onClickLoadingCancelButton(cancelRequest) {
		this.loadingCancelButton.addEventListener('click', cancelRequest)
	}

	updateLoadingMessage() {
		// this.loadingMessage.textContent = 'Loading...'

		setTimeout(() => {
			this.loadingMessage.textContent = 'Recognition in progress...'
		}, 2000)
	}

	hideModalOverlay() {
		this.modalOverlay.classList.remove('modal--show')
	}

	showLoadingModal() {
		this.loadingMessage.textContent = 'Loading...'
		this.hideErrorModal()
		this.hideAddNewFileModal()
		this.updateLoadingMessage() // ????????????
		this.loadingModal.classList.remove('modal--hide')
	}

	showAddNewFileModal() {
		this.hideErrorModal()
		this.showModalOverlay()
		this.hideLoadingModal()
		this.addNewFileModal.classList.remove('modal--hide')
	}

	hideAddNewFileModal() {
		this.addNewFileModal.classList.add('modal--hide')
	}

	hideLoadingModal() {
		this.loadingModal.classList.add('modal--hide')
	}

	showErrorModal(message) {
		this.hideLoadingModal()
		this.errorMessage.textContent = message
		this.errorModal.classList.remove('modal--hide')
	}

	hideErrorModal() {
		this.errorModal.classList.add('modal--hide')
	}
}
