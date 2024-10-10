import {ImageDocumentHandler} from "./_image-document-handler.mjs"
import {JsonDocumentHandler} from "./_json-document-handler.mjs"

export class DocumentManagementFacade {
    constructor() {
        this.imageManagement = new ImageDocumentHandler()
        this.jsonManagement = new JsonDocumentHandler()
    }
    handleFileInfo(file) {
        this.imageManagement.handleFileImage(file)
        this.jsonManagement.handleFileJsonInfo(file)
    }
}