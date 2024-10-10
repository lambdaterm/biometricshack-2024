export class JsonDocumentHandler {
    constructor() {
        this.imageBase64 = document.getElementById('document-image-base64')
        this.documentText = document.getElementById('document-text')
        this.saveButton = document.getElementById('save-button')
        this.copyButton = document.getElementById('copy-button')
        this.textNotFound = document.querySelector('.document-text-empty')

        this.saveButton.addEventListener('click', () => this.saveData())
        this.copyButton.addEventListener('click', () => this.copyText())
    }
    handleFileJsonInfo(file) {
        this.resetTextNotFoundWarning()
        // const fileItem = file[0]
        // this.documentText.textContent = fileItem.name
    }
    showTextNotFoundWarning() {
        this.textNotFound.classList.add('show-warning')
    }
    resetTextNotFoundWarning() {
        this.textNotFound.classList.remove('show-warning')
    }
    saveData() {
        // Save text
        const text = this.documentText.textContent
        const blob = new Blob([text], { type: "text/plain;charset=utf-8" })
        const link = document.createElement("a")
        link.href = URL.createObjectURL(blob)
        link.download = "savedText.txt"
        link.click()

        // Save image
        const imageSrc = this.imageBase64.src
        const base64Image = imageSrc.split(',')[1]
        const byteCharacters = atob(base64Image)
        const byteArrays = []

        for (let i = 0; i < byteCharacters.length; i++) {
            byteArrays.push(byteCharacters.charCodeAt(i))
        }

        const byteArray = new Uint8Array(byteArrays)
        const imageBlob = new Blob([byteArray], { type: "image/png" })
        const imageLink = document.createElement("a")
        imageLink.href = URL.createObjectURL(imageBlob)
        imageLink.download = "savedImage.png"
        imageLink.click()
    }

    copyText() {
        const text = this.documentText.textContent
        const textarea = document.createElement("textarea")
        textarea.textContent = text
        document.body.appendChild(textarea)
        textarea.select()
        document.execCommand("copy")
        document.body.removeChild(textarea)
        alert('Текст скопирован в буфер обмена!')

        // if (this.documentText.hasAttribute('src')) {
        //     alert('Содержимое не является текстом. Невозможно скопировать изображение.')
        // } else {
        //     alert('Текст скопирован в буфер обмена!')
        // }
    }
}