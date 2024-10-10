export class ImageDocumentHandler {
    constructor() {
        this.documentName = document.getElementById('document-name')
        this.image = document.getElementById('document-image')
    }
    handleFileImage(file) {
        const fileItem = file[0]
        this.documentName.textContent = fileItem.name
        this.image.src = URL.createObjectURL(fileItem)
    }
}