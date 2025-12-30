import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision'
import * as THREE from 'three'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
import './style.css'

const app = document.querySelector('#app')

app.innerHTML = `
  <div class="scene">
    <div class="backdrop"></div>
    <div class="stage">
      <div class="viewport">
        <canvas id="three"></canvas>
        <video id="video" autoplay muted playsinline></video>
        <canvas id="overlay"></canvas>
      </div>
      <div id="message" class="message" aria-live="polite"></div>
    </div>
  </div>
`

const video = document.querySelector('#video')
const viewport = document.querySelector('.viewport')
const threeCanvas = document.querySelector('#three')
const canvas = document.querySelector('#overlay')
const ctx = canvas.getContext('2d')
const messageEl = document.querySelector('#message')

let faceLandmarker
let lastVideoTime = -1

let renderer
let scene
let camera
let model
let modelGroup
let modelPivot
let modelBaseSize = 1
let modelReady = false
const modelState = {
  position: new THREE.Vector3(),
  quaternion: new THREE.Quaternion(),
  scale: 1
}

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
  const displayWidth = viewport.clientWidth
  const displayHeight = viewport.clientHeight
  if (!displayWidth || !displayHeight) return

  const dpr = window.devicePixelRatio || 1
  canvas.width = Math.round(displayWidth * dpr)
  canvas.height = Math.round(displayHeight * dpr)
  canvas.style.width = `${displayWidth}px`
  canvas.style.height = `${displayHeight}px`
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  ctx.lineJoin = 'round'
  ctx.lineCap = 'round'

  if (renderer && camera) {
    renderer.setPixelRatio(dpr)
    renderer.setSize(displayWidth, displayHeight, false)
    camera.left = -displayWidth / 2
    camera.right = displayWidth / 2
    camera.top = displayHeight / 2
    camera.bottom = -displayHeight / 2
    camera.updateProjectionMatrix()
  }
}

function getCoverTransform() {
  const displayWidth = viewport.clientWidth
  const displayHeight = viewport.clientHeight
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

function getCenter(points) {
  const total = points.reduce(
    (acc, point) => {
      acc.x += point.x
      acc.y += point.y
      return acc
    },
    { x: 0, y: 0 }
  )
  return {
    x: total.x / points.length,
    y: total.y / points.length
  }
}

function getCenter3D(landmarks, indices) {
  let count = 0
  const total = { x: 0, y: 0, z: 0 }
  for (const index of indices) {
    const point = landmarks[index]
    if (!point) continue
    total.x += point.x
    total.y += point.y
    total.z += point.z ?? 0
    count += 1
  }
  if (!count) return null
  return {
    x: total.x / count,
    y: total.y / count,
    z: total.z / count
  }
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

function getLandmarkBounds(landmarks, transform) {
  if (!landmarks?.length || !transform) return null
  const min = { x: Infinity, y: Infinity }
  const max = { x: -Infinity, y: -Infinity }
  for (const point of landmarks) {
    const x =
      point.x * transform.videoWidth * transform.scale + transform.offsetX
    const y =
      point.y * transform.videoHeight * transform.scale + transform.offsetY
    min.x = Math.min(min.x, x)
    min.y = Math.min(min.y, y)
    max.x = Math.max(max.x, x)
    max.y = Math.max(max.y, y)
  }
  return {
    min,
    max,
    centerX: (min.x + max.x) / 2,
    centerY: (min.y + max.y) / 2,
    width: max.x - min.x,
    height: max.y - min.y
  }
}

function averageNormalizedPoints(points) {
  let count = 0
  const total = { x: 0, y: 0, z: 0 }
  for (const point of points) {
    if (!point) continue
    total.x += point.x
    total.y += point.y
    total.z += point.z ?? 0
    count += 1
  }
  if (!count) return null
  return {
    x: total.x / count,
    y: total.y / count,
    z: total.z / count
  }
}

function normalizedToWorld(
  point,
  transform,
  zScale,
  viewportWidth,
  viewportHeight
) {
  const x = point.x * transform.videoWidth * transform.scale + transform.offsetX
  const y = point.y * transform.videoHeight * transform.scale + transform.offsetY
  const z = (point.z ?? 0) * zScale
  return new THREE.Vector3(
    x - viewportWidth / 2,
    viewportHeight / 2 - y,
    -z
  )
}

function getFaceQuaternion({
  leftEye,
  rightEye,
  forehead,
  chin,
  transform,
  zScale,
  viewportWidth,
  viewportHeight
}) {
  if (!leftEye || !rightEye || !forehead || !chin) return null
  const left = normalizedToWorld(
    leftEye,
    transform,
    zScale,
    viewportWidth,
    viewportHeight
  )
  const right = normalizedToWorld(
    rightEye,
    transform,
    zScale,
    viewportWidth,
    viewportHeight
  )
  const upPoint = normalizedToWorld(
    forehead,
    transform,
    zScale,
    viewportWidth,
    viewportHeight
  )
  const downPoint = normalizedToWorld(
    chin,
    transform,
    zScale,
    viewportWidth,
    viewportHeight
  )

  const xAxis = right.sub(left).normalize()
  const upAxis = upPoint.sub(downPoint).normalize()

  if (xAxis.lengthSq() < 1e-6 || upAxis.lengthSq() < 1e-6) return null

  const zAxis = new THREE.Vector3().crossVectors(xAxis, upAxis).normalize()
  const correctedUp = new THREE.Vector3().crossVectors(zAxis, xAxis).normalize()

  const basis = new THREE.Matrix4().makeBasis(xAxis, correctedUp, zAxis)
  return new THREE.Quaternion().setFromRotationMatrix(basis)
}

function drawFeatureWindow(
  points,
  transform,
  { contentScale = 1, contentOffsetX = 0, contentOffsetY = 0 } = {}
) {
  if (!points || points.length < 3) return
  const center = getCenter(points)
  ctx.save()
  ctx.beginPath()
  ctx.moveTo(points[0].x, points[0].y)
  for (let i = 1; i < points.length; i += 1) {
    ctx.lineTo(points[i].x, points[i].y)
  }
  ctx.closePath()
  ctx.clip()
  ctx.translate(center.x + contentOffsetX, center.y + contentOffsetY)
  ctx.scale(contentScale, contentScale)
  ctx.translate(-center.x, -center.y)
  ctx.drawImage(
    video,
    transform.offsetX,
    transform.offsetY,
    transform.videoWidth * transform.scale,
    transform.videoHeight * transform.scale
  )
  ctx.restore()
}

function initThree() {
  renderer = new THREE.WebGLRenderer({
    canvas: threeCanvas,
    antialias: true,
    alpha: false
  })
  renderer.setClearColor(0x000000, 1)
  renderer.outputColorSpace = THREE.SRGBColorSpace
  renderer.toneMapping = THREE.ACESFilmicToneMapping
  renderer.toneMappingExposure = 1.1

  scene = new THREE.Scene()
  camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 2000)
  camera.position.z = 800
  modelGroup = new THREE.Group()
  modelPivot = new THREE.Group()
  modelGroup.add(modelPivot)
  scene.add(modelGroup)

  const hemi = new THREE.HemisphereLight(0xfff2d4, 0x1b1b1b, 0.6)
  scene.add(hemi)
  const key = new THREE.DirectionalLight(0xfff0d8, 1.2)
  key.position.set(200, 300, 400)
  scene.add(key)
  const rim = new THREE.DirectionalLight(0xffa95a, 0.6)
  rim.position.set(-200, 120, -300)
  scene.add(rim)

  const loader = new GLTFLoader()
  loader.load(
    '/orange.glb',
    (gltf) => {
      model = gltf.scene
      model.traverse((child) => {
        if (!child.isMesh) return
        const materials = Array.isArray(child.material)
          ? child.material
          : [child.material]
        materials.forEach((material) => {
          if (material?.map) {
            material.map.colorSpace = THREE.SRGBColorSpace
          } else if (material?.color) {
            material.color.set('#ff8a1f')
          }
          if (material) {
            material.metalness = 0.1
            material.roughness = 0.6
            material.needsUpdate = true
          }
        })
      })

      const box = new THREE.Box3().setFromObject(model)
      const size = new THREE.Vector3()
      box.getSize(size)
      modelBaseSize = Math.max(size.x, size.y, size.z) || 1
      const center = new THREE.Vector3()
      box.getCenter(center)
      model.position.sub(center)
      modelPivot.add(model)
      modelReady = true
    },
    undefined,
    () => {
      setMessage('Failed to load the orange model.')
    }
  )
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
    initThree()
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

    if (results.faceLandmarks?.length) {
      const transform = getCoverTransform()
      if (transform) {
        for (const landmarks of results.faceLandmarks) {
          const leftEyePoints = getLoopPoints(landmarks, leftEyeLoop, transform)
          const rightEyePoints = getLoopPoints(
            landmarks,
            rightEyeLoop,
            transform
          )
          const mouthPoints = getLoopPoints(landmarks, mouthLoop, transform)

          if (leftEyePoints && rightEyePoints) {
            drawFeatureWindow(leftEyePoints, transform)
            drawFeatureWindow(rightEyePoints, transform)
            if (mouthPoints) drawFeatureWindow(mouthPoints, transform)

            if (modelReady && modelGroup) {
              const bounds = getLandmarkBounds(landmarks, transform)
              if (!bounds) continue

              const viewportWidth = viewport.clientWidth
              const viewportHeight = viewport.clientHeight
              const faceWidth = bounds.width
              const targetScale =
                (faceWidth * 1.8) / (modelBaseSize || 1)

              const leftEyeCenter = getCenter(leftEyePoints)
              const rightEyeCenter = getCenter(rightEyePoints)
              const eyeDx = rightEyeCenter.x - leftEyeCenter.x
              const eyeDy = rightEyeCenter.y - leftEyeCenter.y
              const roll = Math.atan2(eyeDy, eyeDx)

              const leftEye3D = getCenter3D(landmarks, leftEyeLoop)
              const rightEye3D = getCenter3D(landmarks, rightEyeLoop)
              const mouth3D = getCenter3D(landmarks, mouthLoop)
              const nose = landmarks[1]
              const forehead = landmarks[10]
              const chin = landmarks[152]
              const zScale = faceWidth || transform.videoWidth * transform.scale

              const anchor = averageNormalizedPoints([
                leftEye3D,
                rightEye3D,
                mouth3D,
                nose
              ])

              const targetPosition = anchor
                ? normalizedToWorld(
                    anchor,
                    transform,
                    zScale,
                    viewportWidth,
                    viewportHeight
                  )
                : new THREE.Vector3(
                    bounds.centerX - viewportWidth / 2,
                    viewportHeight / 2 - bounds.centerY,
                    0
                  )
              targetPosition.z = 0

              const faceQuaternion = getFaceQuaternion({
                leftEye: leftEye3D,
                rightEye: rightEye3D,
                forehead,
                chin,
                transform,
                zScale,
                viewportWidth,
                viewportHeight
              })

              const targetQuaternion = faceQuaternion
                ? faceQuaternion
                : new THREE.Quaternion().setFromEuler(
                    new THREE.Euler(0, 0, roll)
                  )

              modelState.position.lerp(
                targetPosition,
                0.2
              )
              modelState.quaternion.slerp(targetQuaternion, 0.2)
              modelState.scale =
                modelState.scale * 0.82 + targetScale * 0.18

              modelGroup.position.copy(modelState.position)
              modelGroup.scale.setScalar(modelState.scale)
              modelGroup.quaternion.copy(modelState.quaternion)
              modelGroup.visible = true
            }
          }
        }
      }
    } else if (modelGroup) {
      modelGroup.visible = false
    }
  }

  if (renderer && scene && camera) {
    renderer.render(scene, camera)
  }

  requestAnimationFrame(predict)
}

window.addEventListener('resize', resizeCanvas)

init()
