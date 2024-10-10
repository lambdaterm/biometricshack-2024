import {Screen} from "./_screen.mjs"
import {UploadFacade} from "./_uploadFasade.mjs"
import {Dialog} from "./_dialog.mjs"
import JSONFormatter from "./json-formatter.js"
import {DocumentManagementFacade} from "./_document-management-facade.mjs"
import {getServiceStatus, performImageProcessing} from "./_service-api.js"

const uploadFacade = new UploadFacade()
const modal = new Dialog()
const documentManagementFacade = new DocumentManagementFacade()

const cancelRequest = () => {
    console.log('Cancel request')
}

function imageToBase64(img) {
    return new Promise((resolve, reject) => {
        const canvas = document.createElement('canvas')
        canvas.width = img.naturalWidth
        canvas.height = img.naturalHeight
        const ctx = canvas.getContext('2d')
        ctx.drawImage(img, 0, 0)
        canvas.toBlob(blob => {
            const reader = new FileReader()
            reader.onloadend = () => resolve(reader.result)
            reader.onerror = reject
            reader.readAsDataURL(blob)
        }, 'image/png')
    })
}

const imageBase64 = document.getElementById('document-image')

const recognizeButton = document.querySelector('.recognize-button')

recognizeButton.addEventListener('click', (event) => {
    // modal.isShowModal(true)

    const img = documentManagementFacade.imageManagement.image
    imageToBase64(img).then(base64Image => {
        const base64Data = base64Image.split(',')[1]
        const requestOptions = {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ image_base64: base64Data })
        }

        performImageProcessing(requestOptions)
            .then(response => response.json())
            .then(data => {
                if (data.Image_base64) {
                    imageBase64.src = `data:image/png;base64,${data.Image_base64}`
                } else {
                    imageBase64.src = ''
                }
                // modal.isShowModal(false)
                if (data.Results) {
                    // let jsonString = data.Results
                    // jsonString = jsonString.replace(/'/g, '"')
                    //
                    // jsonString = jsonString.replace(/array\((.*?)\), dtype=uint8/g, "[$1]")
                    // jsonString = jsonString.replace(/True/g, 'true')
                    // const resultsObject = JSON.parse(jsonString)
                    const formatter = new JSONFormatter(data.Results)
                    const documentTextElement = documentManagementFacade.jsonManagement.documentText
                    documentTextElement.appendChild(formatter.render())
                } else {
                    documentManagementFacade.jsonManagement.showTextNotFoundWarning()
                }
            })
            .catch(error => {
                console.error('Ошибка при получении данных:', error)
                modal.showErrorModal(error.message)
            })
    }).catch(error => {
        console.error('Ошибка при преобразовании изображения:', error)
        modal.showErrorModal(error.message)
    })

    //     const formData = new FormData()
    //     formData.append('image_base64', documentManagementFacade.imageManagement.image)
    //     const requestOptions = {
    //         method: 'POST',
    //         // headers: {
    //         //     'Content-Type': 'multipart/form-data'
    //         // },
    //         body: formData
    //     }
    //
    //     performImageProcessing(requestOptions)
    //         .then(response => response.json())
    //         .then(data => {
    //             console.log(data)
    //             modal.isShowModal(false)
    //             if (data.text) {
    //                 const formatter = new JSONFormatter(data)
    //                 const documentTextElement = documentManagementFacade.jsonManagement.documentText
    //                 documentTextElement.appendChild(formatter.render())
    //             } else {
    //                 documentManagementFacade.jsonManagement.showTextNotFoundWarning()
    //             }
    //         })
    //         .catch(error => {
    //             console.error('Ошибка при получении данных:', error)
    //             modal.showErrorModal(error.message)
    //         })
})

const handleFileUpload = (file) => {
    imageBase64.src = ''
    documentManagementFacade.jsonManagement.documentText.textContent = ''
    if (file.length) {
        // modal.isShowModal(true)
        documentManagementFacade.handleFileInfo(file)
        modal.isShowModal(false)
        screen2.show()
        // setTimeout(() => {
        //     modal.isShowModal(false)
        //     screen2.show()
        // }, 4000)
    }
}

uploadFacade.linkUploadFileToHandlers(handleFileUpload)

modal.onClickLoadingCancelButton(cancelRequest)

const screen1 = new Screen('screen1')
const screen2 = new Screen('screen2')

screen1.show()
screen2.hide()
// screen2.show()

const backToScreen1Button = document.querySelector('.navigation-back-button')
backToScreen1Button.addEventListener('click', (event) => {
    if (event.target) {
        screen2.hide()
        screen1.show()
    }
})

const checkServiceStatus = () => {
    const statusIndicators = document.querySelectorAll('.status-indicator')
    getServiceStatus()
        .then(response => {
            statusIndicators.forEach(indicator => {
                if (response.ok) {
                    indicator.style.backgroundColor = 'green'
                } else {
                    indicator.style.backgroundColor = '#c80202'
                }
            })
        })
        .catch(error => {
            statusIndicators.forEach(indicator => {
                indicator.style.backgroundColor = '#c80202'
            })
            console.error('Ошибка:', error)
        })
}

setInterval(() => {
    checkServiceStatus()
}, 3000)
