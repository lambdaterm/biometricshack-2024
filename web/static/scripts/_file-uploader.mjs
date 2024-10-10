export class FileUploader {
    constructor() {
        this.uploadFile = null
        this.addNewFileButton = document.getElementById('add-new-file')
        this.uploadInputButton = document.getElementById('upload-input')
        this.addNewFileModalButton = document.querySelector('.add-new-file-modal-button')
        this.uploadBroseButton = document.getElementById('brose-button')

        this.addNewFileButton.addEventListener('click', (event) => this.showUploadModal(event))
        this.addNewFileModalButton.addEventListener('click', () => this.showUploadModal())
        this.uploadInputButton.addEventListener('change', (event) => this.addNewFile(event))
        this.uploadBroseButton.addEventListener('change', (event) => this.addNewFile(event))
    }
    addNewFile(event) {
        const files = event.target.files
        this.uploadFile(files)
    }
    showUploadModal() {
        this.uploadInputButton.click()
    }
}