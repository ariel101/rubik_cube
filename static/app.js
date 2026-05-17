async function updateStatus() {

    const response = await fetch('/status')

    const data = await response.json()

    document.getElementById('face').innerText =
        data.current_face

    document.getElementById('detected').innerText =
        data.detected

    document.getElementById('expected').innerText =
        data.expected

    document.getElementById('solution').innerText =
        data.solution || 'Esperando...'
}

async function captureFace() {

    await fetch('/capture', {
        method: 'POST'
    })
}

async function resetCube() {

    await fetch('/reset', {
        method: 'POST'
    })
}

setInterval(updateStatus, 500)