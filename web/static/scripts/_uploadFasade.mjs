import { DragAndDrop } from "./_drag-and-drop.mjs"
import { FileUploader } from "./_file-uploader.mjs"

export class UploadFacade  {
    constructor() {
        this.modal = new FileUploader ()
        this.dragAndDrop = new DragAndDrop()
    }
    linkUploadFileToHandlers(uploadFile) {
        this.modal.uploadFile = uploadFile
        this.dragAndDrop.uploadFile = uploadFile
    }
}