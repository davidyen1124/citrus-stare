import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision'
import './style.css'

const app = document.querySelector('#app')

app.innerHTML = `
  <div class="scene">
    <div class="backdrop"></div>
    <div class="stage">
      <div class="viewport">
        <video id="video" autoplay muted playsinline></video>
        <canvas id="overlay"></canvas>
      </div>
      <div id="message" class="message" aria-live="polite"></div>
    </div>
  </div>
`

const video = document.querySelector('#video')
const canvas = document.querySelector('#overlay')
const ctx = canvas.getContext('2d')
const messageEl = document.querySelector('#message')

let faceLandmarker
let lastVideoTime = -1

const modelAssetPath =
  'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'
const wasmPath =
  'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'
const leftEyeConnections = FaceLandmarker.FACE_LANDMARKS_LEFT_EYE ?? []
const rightEyeConnections = FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE ?? []
const mouthConnections = FaceLandmarker.FACE_LANDMARKS_LIPS ?? []
const leftEyeLoop = buildLoop(leftEyeConnections)
const rightEyeLoop = buildLoop(rightEyeConnections)
const mouthLoop = buildLoop(mouthConnections)

function setMessage(text = '') {
  messageEl.textContent = text
  messageEl.dataset.visible = text ? 'true' : 'false'
}

function resizeCanvas() {
  const displayWidth = video.clientWidth
  const displayHeight = video.clientHeight
  if (!displayWidth || !displayHeight) return

  const dpr = window.devicePixelRatio || 1
  canvas.width = Math.round(displayWidth * dpr)
  canvas.height = Math.round(displayHeight * dpr)
  canvas.style.width = `${displayWidth}px`
  canvas.style.height = `${displayHeight}px`
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  ctx.lineJoin = 'round'
  ctx.lineCap = 'round'
}

function getCoverTransform() {
  const displayWidth = canvas.clientWidth
  const displayHeight = canvas.clientHeight
  const videoWidth = video.videoWidth
  const videoHeight = video.videoHeight
  if (!displayWidth || !displayHeight || !videoWidth || !videoHeight) return null

  const scale = Math.max(displayWidth / videoWidth, displayHeight / videoHeight)
  const offsetX = (displayWidth - videoWidth * scale) / 2
  const offsetY = (displayHeight - videoHeight * scale) / 2

  return { scale, offsetX, offsetY, videoWidth, videoHeight }
}

function buildLoop(connections) {
  if (!connections || typeof connections[Symbol.iterator] !== 'function') return []
  const adjacency = new Map()
  for (const connection of connections) {
    const start =
      connection?.start ?? (Array.isArray(connection) ? connection[0] : null)
    const end =
      connection?.end ?? (Array.isArray(connection) ? connection[1] : null)
    if (start == null || end == null) continue
    if (!adjacency.has(start)) adjacency.set(start, new Set())
    if (!adjacency.has(end)) adjacency.set(end, new Set())
    adjacency.get(start).add(end)
    adjacency.get(end).add(start)
  }
  if (!adjacency.size) return []

  const start = adjacency.keys().next().value
  const loop = [start]
  let previous = null
  let current = start

  while (true) {
    const neighbors = Array.from(adjacency.get(current) ?? [])
    if (!neighbors.length) break
    const next = neighbors.find((node) => node !== previous) ?? neighbors[0]
    if (next === start) break
    if (loop.includes(next)) break
    loop.push(next)
    previous = current
    current = next
    if (loop.length > adjacency.size + 2) break
  }

  return loop
}

function getLoopPoints(landmarks, loop, transform) {
  if (!loop || loop.length < 3) return null
  const points = []
  for (const index of loop) {
    const point = landmarks[index]
    if (!point) return null
    points.push({
      x: point.x * transform.videoWidth * transform.scale + transform.offsetX,
      y: point.y * transform.videoHeight * transform.scale + transform.offsetY
    })
  }
  return points
}

function drawEyeWindow(
  landmarks,
  loop,
  transform,
  { strokeStyle, lineWidth, drawOutline = true }
) {
  const points = getLoopPoints(landmarks, loop, transform)
  if (!points) return

  ctx.save()
  ctx.beginPath()
  ctx.moveTo(points[0].x, points[0].y)
  for (let i = 1; i < points.length; i += 1) {
    ctx.lineTo(points[i].x, points[i].y)
  }
  ctx.closePath()
  ctx.clip()
  ctx.drawImage(
    video,
    transform.offsetX,
    transform.offsetY,
    transform.videoWidth * transform.scale,
    transform.videoHeight * transform.scale
  )
  ctx.restore()

  if (drawOutline) {
    ctx.beginPath()
    ctx.moveTo(points[0].x, points[0].y)
    for (let i = 1; i < points.length; i += 1) {
      ctx.lineTo(points[i].x, points[i].y)
    }
    ctx.closePath()
    ctx.strokeStyle = strokeStyle
    ctx.lineWidth = lineWidth
    ctx.stroke()
  }
}

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error('Camera API not supported in this browser.')
  }

  const stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: {
      facingMode: 'user',
      width: { ideal: 1280 },
      height: { ideal: 720 }
    }
  })

  video.srcObject = stream
  await new Promise((resolve) => {
    video.onloadedmetadata = () => resolve()
  })
  await video.play()
  resizeCanvas()
}

function describeCameraError(error) {
  if (!error) return 'Unable to access the camera.'
  if (error.name === 'NotAllowedError') {
    return 'Camera access was blocked. Please allow permission and refresh.'
  }
  if (error.name === 'NotFoundError') {
    return 'No camera was found on this device.'
  }
  if (error.name === 'NotReadableError') {
    return 'Your camera is already in use by another app.'
  }
  if (error.name === 'OverconstrainedError') {
    return 'The requested camera constraints are not supported.'
  }
  return null
}

function describeInitError(error) {
  const cameraMessage = describeCameraError(error)
  if (cameraMessage) return cameraMessage
  if (error?.message?.toLowerCase().includes('fetch')) {
    return 'Failed to load MediaPipe assets. Check your connection and refresh.'
  }
  return error?.message || 'Unable to start face tracking.'
}

async function setupFaceLandmarker() {
  setMessage('Loading model…')
  const resolver = await FilesetResolver.forVisionTasks(wasmPath)
  faceLandmarker = await FaceLandmarker.createFromOptions(resolver, {
    baseOptions: {
      modelAssetPath,
      delegate: 'GPU'
    },
    runningMode: 'VIDEO',
    numFaces: 1
  })
}

async function init() {
  try {
    setMessage('Allow camera access to begin.')
    await setupFaceLandmarker()
    await startCamera()
    setMessage('')
    requestAnimationFrame(predict)
  } catch (error) {
    console.error(error)
    setMessage(describeInitError(error))
  }
}

function predict() {
  if (!faceLandmarker || video.readyState < 2) {
    requestAnimationFrame(predict)
    return
  }

  const now = performance.now()
  if (video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime
    const results = faceLandmarker.detectForVideo(video, now)

    const dpr = window.devicePixelRatio || 1
    ctx.clearRect(0, 0, canvas.width / dpr, canvas.height / dpr)
    ctx.fillStyle = '#000000'
    ctx.fillRect(0, 0, canvas.width / dpr, canvas.height / dpr)

    if (results.faceLandmarks?.length) {
      const transform = getCoverTransform()
      if (transform) {
        for (const landmarks of results.faceLandmarks) {
          drawEyeWindow(landmarks, leftEyeLoop, transform, {
            strokeStyle: 'rgba(124, 255, 214, 0.95)',
            lineWidth: 2,
            drawOutline: false
          })
          drawEyeWindow(landmarks, rightEyeLoop, transform, {
            strokeStyle: 'rgba(120, 192, 255, 0.95)',
            lineWidth: 2,
            drawOutline: false
          })
          drawEyeWindow(landmarks, mouthLoop, transform, {
            strokeStyle: 'rgba(255, 168, 120, 0.95)',
            lineWidth: 2,
            drawOutline: false
          })
        }
      }
    }
  }

  requestAnimationFrame(predict)
}

window.addEventListener('resize', resizeCanvas)

init()
