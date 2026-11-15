import Network
import XCTest

final class PhoneBridge: XCTestCase {
  func testBridge() throws {
    let env = ProcessInfo.processInfo.environment
    let seconds = Double(env["BRIDGE_SECONDS"] ?? "600") ?? 600
    let listenPort = NWEndpoint.Port(rawValue: UInt16(env["BRIDGE_LISTEN"] ?? "18443")!)!
    let tunnelPort = NWEndpoint.Port(rawValue: UInt16(env["BRIDGE_TUNNEL"] ?? "18444")!)!
    let bundle = env["BRIDGE_BUNDLE"] ?? ""
    let queue = DispatchQueue(label: "bridge", attributes: .concurrent)
    let stats = Stats()
    let pool = Pool()

    let tunnel = try NWListener(using: loopback(), on: tunnelPort)
    tunnel.stateUpdateHandler = { state in print("BRIDGE tunnel \(state)") }
    tunnel.newConnectionHandler = { connection in
      connection.stateUpdateHandler = { state in
        switch state {
        case .failed, .cancelled: pool.remove(connection)
        default: break
        }
      }
      connection.start(queue: queue)
      pool.add(connection)
    }
    tunnel.start(queue: queue)

    let listener = try NWListener(using: loopback(), on: listenPort)
    listener.stateUpdateHandler = { state in print("BRIDGE listener \(state)") }
    listener.newConnectionHandler = { inbound in
      queue.async {
        guard let outbound = pool.take(timeout: 10) else {
          print("BRIDGE no idle tunnel connection")
          inbound.cancel()
          return
        }
        stats.add(opened: 1)
        outbound.stateUpdateHandler = { state in
          switch state {
          case .failed, .cancelled: inbound.cancel()
          default: break
          }
        }
        inbound.stateUpdateHandler = { state in
          switch state {
          case .failed, .cancelled: outbound.cancel()
          default: break
          }
        }
        pump(from: inbound, to: outbound, stats: stats)
        pump(from: outbound, to: inbound, stats: stats)
        inbound.start(queue: queue)
      }
    }
    listener.start(queue: queue)

    let app = bundle.isEmpty ? nil : XCUIApplication(bundleIdentifier: bundle)
    app?.activate()
    let command = URL(string: "http://127.0.0.1:\(listenPort.rawValue)/__bridge/command")!
    let start = Date()
    var lastPrint = -10.0
    var lastFront = 0.0
    while Date().timeIntervalSince(start) < seconds {
      let elapsed = Date().timeIntervalSince(start)
      if let app, elapsed - lastFront >= 2 {
        lastFront = elapsed
        if app.state != .runningForeground { app.activate() }
      }
      if let app, pool.count() > 0 {
        let order = fetchCommand(command)
        if order == "tap" {
          print("BRIDGE tap")
          app.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.5)).tap()
        } else if order.hasPrefix("home ") {
          let away = Double(order.dropFirst(5)) ?? 10
          print("BRIDGE home for \(away) s")
          XCUIDevice.shared.press(.home)
          Thread.sleep(forTimeInterval: away)
          app.activate()
          print("BRIDGE back in front")
          lastFront = Date().timeIntervalSince(start)
        }
      }
      if elapsed - lastPrint >= 5 {
        lastPrint = elapsed
        let state = app.map { String($0.state.rawValue) } ?? "none"
        let thermal = ProcessInfo.processInfo.thermalState.rawValue
        print("BRIDGE \(Int(elapsed)) s, thermal \(thermal), \(stats.text()), idle \(pool.count()), app \(state)")
      }
      Thread.sleep(forTimeInterval: 1)
    }
    listener.cancel()
    tunnel.cancel()
  }
}

func fetchCommand(_ url: URL) -> String {
  var request = URLRequest(url: url)
  request.timeoutInterval = 2
  let done = DispatchSemaphore(value: 0)
  var order = ""
  URLSession.shared.dataTask(with: request) { data, _, _ in
    if let data, let text = String(data: data, encoding: .utf8) {
      order = text.trimmingCharacters(in: .whitespacesAndNewlines)
    }
    done.signal()
  }.resume()
  _ = done.wait(timeout: .now() + 3)
  return order
}

func loopback() -> NWParameters {
  let parameters = NWParameters.tcp
  parameters.requiredInterfaceType = .loopback
  parameters.allowLocalEndpointReuse = true
  return parameters
}

final class Pool {
  private let lock = NSCondition()
  private var idle: [NWConnection] = []
  func add(_ connection: NWConnection) {
    lock.lock(); idle.append(connection); lock.signal(); lock.unlock()
  }
  func remove(_ connection: NWConnection) {
    lock.lock(); idle.removeAll { $0 === connection }; lock.unlock()
  }
  func take(timeout: TimeInterval) -> NWConnection? {
    let deadline = Date().addingTimeInterval(timeout)
    lock.lock(); defer { lock.unlock() }
    while idle.isEmpty {
      if !lock.wait(until: deadline) { return nil }
    }
    return idle.removeFirst()
  }
  func count() -> Int {
    lock.lock(); defer { lock.unlock() }
    return idle.count
  }
}

final class Stats {
  private let lock = NSLock()
  private var opened = 0
  private var bytes = 0
  func add(opened: Int = 0, bytes: Int = 0) {
    lock.lock(); self.opened += opened; self.bytes += bytes; lock.unlock()
  }
  func text() -> String {
    lock.lock(); defer { lock.unlock() }
    return "connections \(opened), bytes \(bytes)"
  }
}

func pump(from source: NWConnection, to target: NWConnection, stats: Stats) {
  source.receive(minimumIncompleteLength: 1, maximumLength: 1 << 16) { data, _, isComplete, error in
    if let data, !data.isEmpty {
      stats.add(bytes: data.count)
      target.send(content: data, completion: .contentProcessed { sendError in
        if sendError != nil { source.cancel(); return }
        if isComplete {
          target.send(content: nil, contentContext: .finalMessage, isComplete: true, completion: .idempotent)
        } else {
          pump(from: source, to: target, stats: stats)
        }
      })
      return
    }
    if isComplete || error != nil {
      target.send(content: nil, contentContext: .finalMessage, isComplete: true, completion: .idempotent)
      return
    }
    pump(from: source, to: target, stats: stats)
  }
}
