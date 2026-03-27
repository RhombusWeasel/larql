use std::collections::HashSet;
use std::path::PathBuf;
use std::time::Instant;

use clap::Args;
use indicatif::{ProgressBar, ProgressStyle};
use larql_core::loader::vector_loader::{
    self, discover_vector_files, LoadCallbacks, LoaderError, TableSummary, VectorReader,
};

#[derive(Args)]
pub struct VectorLoadArgs {
    /// Directory containing .vectors.jsonl files from vector-extract.
    input: PathBuf,

    /// SurrealDB endpoint (e.g. http://localhost:8000).
    #[arg(long, default_value = "http://localhost:8000")]
    surreal: String,

    /// SurrealDB namespace.
    #[arg(long)]
    ns: String,

    /// SurrealDB database.
    #[arg(long)]
    db: String,

    /// SurrealDB username.
    #[arg(long, default_value = "root")]
    user: String,

    /// SurrealDB password.
    #[arg(long, default_value = "root")]
    pass: String,

    /// Tables to load (comma-separated). Default: all found in input dir.
    #[arg(long, value_delimiter = ',')]
    tables: Option<Vec<String>>,

    /// Layers to load (comma-separated). Default: all.
    #[arg(long, value_delimiter = ',')]
    layers: Option<Vec<usize>>,

    /// Batch size for INSERT transactions.
    #[arg(long, default_value = "500")]
    batch_size: usize,

    /// Resume interrupted load (skips completed layers).
    #[arg(long)]
    resume: bool,

    /// Create schema only (no data load).
    #[arg(long)]
    schema_only: bool,
}

/// HTTP client wrapper for SurrealDB's /sql endpoint.
struct SurrealClient {
    client: reqwest::blocking::Client,
    url: String,
    ns: String,
    db: String,
}

impl SurrealClient {
    fn new(url: &str, ns: &str, db: &str, user: &str, pass: &str) -> Self {
        // Strip trailing slash
        let url = url.trim_end_matches('/').to_string();

        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(300))
            .default_headers({
                let mut headers = reqwest::header::HeaderMap::new();
                let auth =
                    base64_encode(&format!("{user}:{pass}"));
                headers.insert(
                    reqwest::header::AUTHORIZATION,
                    format!("Basic {auth}").parse().unwrap(),
                );
                headers
            })
            .build()
            .expect("failed to build HTTP client");

        Self {
            client,
            url,
            ns: ns.to_string(),
            db: db.to_string(),
        }
    }

    /// Execute a SurQL query and return the raw JSON response.
    fn query(&self, sql: &str) -> Result<serde_json::Value, LoaderError> {
        let resp = self
            .client
            .post(&format!("{}/sql", self.url))
            .header("surreal-ns", &self.ns)
            .header("surreal-db", &self.db)
            .header("Accept", "application/json")
            .body(sql.to_string())
            .send()
            .map_err(|e| LoaderError::Surreal(format!("HTTP error: {e}")))?;

        let status = resp.status();
        let body = resp
            .text()
            .map_err(|e| LoaderError::Surreal(format!("read error: {e}")))?;

        if !status.is_success() {
            return Err(LoaderError::Surreal(format!(
                "HTTP {status}: {body}"
            )));
        }

        let json: serde_json::Value = serde_json::from_str(&body)
            .map_err(|e| LoaderError::Surreal(format!("JSON parse error: {e}: {body}")))?;

        Ok(json)
    }

    /// Execute SQL, check for errors in the response array.
    fn exec(&self, sql: &str) -> Result<(), LoaderError> {
        let resp = self.query(sql)?;

        // SurrealDB returns an array of results. Check each for errors.
        if let Some(arr) = resp.as_array() {
            for result in arr {
                if let Some(status) = result.get("status").and_then(|s| s.as_str()) {
                    if status == "ERR" {
                        let detail = result
                            .get("result")
                            .and_then(|r| r.as_str())
                            .unwrap_or("unknown error");
                        return Err(LoaderError::Surreal(detail.to_string()));
                    }
                }
            }
        }

        Ok(())
    }

    /// Query completed layers from load_progress table.
    fn completed_layers(&self, table: &str) -> Result<HashSet<usize>, LoaderError> {
        let sql = vector_loader::completed_layers_sql(table);
        let resp = self.query(&sql)?;

        let mut layers = HashSet::new();
        if let Some(arr) = resp.as_array() {
            for result in arr {
                if let Some(rows) = result.get("result").and_then(|r| r.as_array()) {
                    for row in rows {
                        if let Some(layer) = row.get("layer").and_then(|l| l.as_u64()) {
                            layers.insert(layer as usize);
                        }
                    }
                }
            }
        }

        Ok(layers)
    }
}

struct ProgressCallbacks {
    bar: ProgressBar,
}

impl LoadCallbacks for ProgressCallbacks {
    fn on_table_start(&mut self, table: &str, total_records: usize) {
        self.bar.set_length(total_records as u64);
        self.bar.set_position(0);
        self.bar
            .set_message(format!("{table}: {total_records} records"));
    }

    fn on_batch_done(&mut self, _table: &str, _batch_num: usize, records_loaded: usize) {
        self.bar.set_position(records_loaded as u64);
    }

    fn on_table_done(&mut self, table: &str, total_loaded: usize, elapsed_ms: f64) {
        self.bar.set_position(total_loaded as u64);
        eprintln!(
            "  {table}: {total_loaded} records loaded ({:.0}s)",
            elapsed_ms / 1000.0,
        );
    }
}

pub fn run(args: VectorLoadArgs) -> Result<(), Box<dyn std::error::Error>> {
    // Discover vector files
    let all_files = discover_vector_files(&args.input)?;
    if all_files.is_empty() {
        return Err(format!("no .vectors.jsonl files found in {}", args.input.display()).into());
    }

    // Filter by requested tables
    let files: Vec<_> = match &args.tables {
        Some(tables) => all_files
            .into_iter()
            .filter(|(name, _)| tables.contains(name))
            .collect(),
        None => all_files,
    };

    let layer_filter: Option<HashSet<usize>> = args.layers.map(|ls| ls.into_iter().collect());

    eprintln!("Connecting to SurrealDB: {}", args.surreal);
    eprintln!("  ns={}, db={}", args.ns, args.db);
    eprintln!(
        "  tables: {}",
        files.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>().join(", ")
    );

    let client = SurrealClient::new(&args.surreal, &args.ns, &args.db, &args.user, &args.pass);

    // Setup namespace and database
    let setup_sql = vector_loader::setup_sql(&args.ns, &args.db);
    client.exec(&setup_sql)?;
    eprintln!("  namespace/database ready");

    // Create progress tracking table
    client.exec(vector_loader::progress_table_sql())?;

    // Create schemas for each table
    for (component, path) in &files {
        let reader = VectorReader::open(path)?;
        let dimension = reader.header().dimension;
        let schema = vector_loader::schema_sql(component, dimension)?;
        client.exec(&schema)?;
        eprintln!("  schema: {component} (dim={dimension})");
    }

    if args.schema_only {
        eprintln!("\nSchema created. No data loaded (--schema-only).");
        return Ok(());
    }

    // Load data
    let overall_start = Instant::now();
    let mut summaries = Vec::new();

    let bar = ProgressBar::new(0);
    bar.set_style(
        ProgressStyle::default_bar()
            .template("{spinner} [{bar:40}] {pos}/{len} {msg}")
            .unwrap(),
    );
    let mut callbacks = ProgressCallbacks { bar };

    for (component, path) in &files {
        let table_start = Instant::now();

        // Check completed layers for resume
        let completed = if args.resume {
            match client.completed_layers(component) {
                Ok(layers) => {
                    if !layers.is_empty() {
                        eprintln!(
                            "\n  {component}: resuming ({} layers already loaded)",
                            layers.len()
                        );
                    }
                    layers
                }
                Err(_) => HashSet::new(), // table might not exist yet
            }
        } else {
            HashSet::new()
        };

        // Read records from NDJSON file
        let mut reader = VectorReader::open(path)?;
        let all_records = reader.read_all(layer_filter.as_ref())?;

        // Group by layer for progress tracking and resume
        let mut layers_map: std::collections::BTreeMap<usize, Vec<_>> =
            std::collections::BTreeMap::new();
        for record in all_records {
            layers_map.entry(record.layer).or_default().push(record);
        }

        // Filter out completed layers
        let pending_layers: Vec<usize> = layers_map
            .keys()
            .filter(|l| !completed.contains(l))
            .copied()
            .collect();

        let total_pending: usize = pending_layers
            .iter()
            .map(|l| layers_map.get(l).map_or(0, |v| v.len()))
            .sum();

        if total_pending == 0 {
            eprintln!("  {component}: all layers already loaded, skipping");
            summaries.push(TableSummary {
                table: component.clone(),
                records_loaded: 0,
                elapsed_secs: 0.0,
            });
            continue;
        }

        callbacks.on_table_start(component, total_pending);
        let mut total_loaded = 0;

        for layer in &pending_layers {
            let records = match layers_map.remove(layer) {
                Some(r) => r,
                None => continue,
            };

            // Batch insert
            let mut batch_num = 0;
            for chunk in records.chunks(args.batch_size) {
                let sql = vector_loader::batch_insert_sql(component, chunk);
                client.exec(&sql)?;
                total_loaded += chunk.len();
                batch_num += 1;
                callbacks.on_batch_done(component, batch_num, total_loaded);
            }

            // Mark layer done in progress table
            let progress_sql =
                vector_loader::mark_layer_done_sql(component, *layer, records.len());
            client.exec(&progress_sql)?;
        }

        let elapsed_ms = table_start.elapsed().as_secs_f64() * 1000.0;
        callbacks.on_table_done(component, total_loaded, elapsed_ms);

        summaries.push(TableSummary {
            table: component.clone(),
            records_loaded: total_loaded,
            elapsed_secs: table_start.elapsed().as_secs_f64(),
        });
    }

    callbacks.bar.finish_and_clear();

    let elapsed = overall_start.elapsed();
    let total: usize = summaries.iter().map(|s| s.records_loaded).sum();

    eprintln!("\nCompleted in {:.1}min", elapsed.as_secs_f64() / 60.0);
    eprintln!("  Total records loaded: {total}");
    for s in &summaries {
        if s.records_loaded > 0 {
            eprintln!(
                "  {}: {} records ({:.0}s)",
                s.table,
                s.records_loaded,
                s.elapsed_secs,
            );
        }
    }

    Ok(())
}

/// Simple base64 encoder for Basic auth (avoids adding a base64 crate).
fn base64_encode(input: &str) -> String {
    const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let bytes = input.as_bytes();
    let mut out = String::with_capacity((bytes.len() + 2) / 3 * 4);

    for chunk in bytes.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
        let triple = (b0 << 16) | (b1 << 8) | b2;

        out.push(CHARS[((triple >> 18) & 0x3F) as usize] as char);
        out.push(CHARS[((triple >> 12) & 0x3F) as usize] as char);

        if chunk.len() > 1 {
            out.push(CHARS[((triple >> 6) & 0x3F) as usize] as char);
        } else {
            out.push('=');
        }

        if chunk.len() > 2 {
            out.push(CHARS[(triple & 0x3F) as usize] as char);
        } else {
            out.push('=');
        }
    }

    out
}
