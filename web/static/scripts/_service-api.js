import {SERVICE_API} from "../constants.js"

export function getServiceStatus() {
    return fetch(`${SERVICE_API}/health`)
}

export function performImageProcessing(requestOptions) {
    return fetch(`${SERVICE_API}/perform_image`, requestOptions)
}