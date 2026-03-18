//! Queue verification: run a quick round-trip test to verify IPC queues work.
//! Helps diagnose Windows issues where host and bootstrapper use different queue mechanisms.

use std::time::Duration;

use interprocess::{QueueFactory, QueueOptions};

const VERIFY_CAPACITY: i64 = 256;

/// Runs a quick round-trip test: spawn a thread that writes then reads, main thread reads then writes.
/// Returns true if the test passed.
pub fn verify_queues_work(prefix: &str) -> bool {
    let verify_in = format!("{}.verify_in", prefix);
    let verify_out = format!("{}.verify_out", prefix);

    let factory = QueueFactory::new();
    let mut main_pub = factory.create_publisher(QueueOptions::with_destroy(
        &verify_out,
        VERIFY_CAPACITY,
        true,
    ));
    let mut main_sub = factory.create_subscriber(QueueOptions::with_destroy(
        &verify_in,
        VERIFY_CAPACITY,
        true,
    ));

    // Spawn thread: write PING to verify_in, read from verify_out (expect PONG)
    let thread_handle = std::thread::spawn({
        let mut thread_pub = factory.create_publisher(QueueOptions::with_destroy(
            &verify_in,
            VERIFY_CAPACITY,
            true,
        ));
        let mut thread_sub = factory.create_subscriber(QueueOptions::with_destroy(
            &verify_out,
            VERIFY_CAPACITY,
            true,
        ));
        move || {
            let _ = thread_pub.try_enqueue(b"PING");
            let mut deadline = std::time::Instant::now() + Duration::from_secs(2);
            while deadline > std::time::Instant::now() {
                if let Some(msg) = thread_sub.try_dequeue() {
                    return msg == b"PONG";
                }
                std::thread::sleep(Duration::from_millis(10));
            }
            false
        }
    });

    // Main: read from verify_in (expect PING), write PONG to verify_out
    let mut deadline = std::time::Instant::now() + Duration::from_secs(2);
    let mut got_ping = false;
    while deadline > std::time::Instant::now() {
        if let Some(msg) = main_sub.try_dequeue() {
            if msg == b"PING" {
                got_ping = true;
                break;
            }
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    if !got_ping {
        logger::warn!("Queue verify: main thread did not receive PING");
        return false;
    }
    if !main_pub.try_enqueue(b"PONG") {
        logger::warn!("Queue verify: main thread failed to enqueue PONG");
        return false;
    }

    match thread_handle.join() {
        Ok(ok) => ok,
        Err(_) => {
            logger::warn!("Queue verify: thread panicked");
            false
        }
    }
}
