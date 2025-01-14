use regex::Regex;
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;


#[derive(Default)]
pub struct Graph {
    // I'm using a BTreeMap which keeps the order of the keys unlike hashmaps
    // this way we can make our program more deterministic
    nodes: BTreeMap<i32, Node>,
}

#[derive(Clone)]
pub struct Node {
    data: String,
    children: Vec<i32>,
    // we will only pay attention to the dims of the matrix
    // the ones resulting from an operation will be infered
    // from e-class analysis in mlir.rs
    rows: i32,
    columns: i32,

    // refers the the position of the mlir value in the map in EqualitySaturationPass.cpp
    // we can think if it as a link to the corresponding mlir value of the result of the current op
    old_id: i32,

    // keeps a link to the the old operation in the map in EqualitySaturationPass.cpp
    // this id can change in few cases 
    // (ex: when applying the rule T(T) = id, id inherits the old op id of the outer transpose)
    old_op_id: i32
}

impl Node {
    pub fn get_data(&self) -> &str {
        &self.data
    }
    pub fn get_children(&self) -> &Vec<i32> {
        &self.children
    }
    pub fn get_rows(&self) -> &i32 {
        &self.rows
    }
    pub fn get_columns(&self) -> &i32 {
        &self.columns
    }
    pub fn get_old_id(&self) -> &i32 {
        &self.old_id
    }
    pub fn get_old_op_id(&self) -> &i32 {
        &self.old_op_id
    }
}

impl Graph {
    // note that for any other op that matrix, rows and columns will be 0
    // we then infer the dimensions using our analysis
    pub fn add_node_new(&mut self, id: i32, data: String, children: Vec<i32>, rows: i32, columns: i32, old_id: i32, old_op_id: i32) {
        self.nodes.insert(id, Node { data, children, rows, columns, old_id, old_op_id});
    }
    pub fn new() -> Self {
        Graph {
            nodes: BTreeMap::new(),
        }
    }
    pub fn get_nodes(&self) -> &BTreeMap<i32, Node> {
        &self.nodes
    }
    pub fn from_file<P>(path: P) -> Self
    where
        P: AsRef<Path>,
    {
        let mut result = Graph::new(); 
        let contents = fs::read_to_string(path).expect("Should have been able to read the file!");

        let mut lines = contents.lines(); // has to be a separate line because we're borrowing it mutably

        // read the first line to figure out what kind of graph
        let first = lines.next().expect("File seems empty...");

        // find the first word of the first line to determine the type of the graph
        // we only support directed graphs for now which serve our purposes
        match first.split_whitespace().next().unwrap_or("") {
            "digraph" => println!("found digraph"),
            _ => panic!("This DOT type isn't supported yet!"),
        }

        // read the rest of the lines until the penultimate one
        // we expect all our node definitions to come first
        let node_regex = 
            Regex::new(r#"^\t(?<id>\d+) \[label="(?<data>.*)", rows=(?<rows>\d+), columns=(?<columns>\d+)\];$"#).unwrap();
        
        for node_string in lines.by_ref().take_while(|&line| !line.trim().is_empty()) {
            // this lets us match for a line of pure split_whitespace
            // lines are of the form (ID -[->] [label="DATA", rows="ROWS", columns="COLUMNS"];).
            // we parse it into a node

            let Some(caps) = node_regex.captures(node_string) else {
                println!("node not found in line '{}'", node_string);
                continue;
            }; 

            let id: i32 = caps["id"].parse().unwrap();
            let data = caps["data"].to_owned();
            let rows = caps["rows"].parse().unwrap();
            let columns = caps["columns"].parse().unwrap();
            result.nodes.insert(
                id,
                Node {
                    data,
                    children: Vec::new(),
                    rows,
                    columns,
                    // old id is the current id in this dot file, 
                    // we store it since the id var will change later in the program after the opts
                    old_id : id, 
                    old_op_id : id      // same with old op id
                },
            );
            println!("node inserted!");
        }
        // then a blank line

        // then all our edges
        let edge_regex = Regex::new(r#"^\t(?<from>\d+) -[->] (?<to>\d+);$"#).unwrap();
        for edge_string in lines {
            // edges look like 'ID -> ID;' since it's a directed graph
            // we keep the matching option for '-' in case we need a non directed graph in the future
            let Some(caps) = edge_regex.captures(edge_string) else {
                continue;
            };

            let from: i32 = caps["from"].parse().unwrap();
            let to: i32 = caps["to"].parse().unwrap();

            let from_node = result.nodes.get_mut(&from).unwrap();
            from_node.children.push(to);
        }

        println!("read graph with {} nodes", result.nodes.len());

        return result;
    }


    pub fn to_file<P>(&self, path: P)
    where
        P: AsRef<Path>,
    {

        // we try here to get rid off the -1 old id value introduced for some new operations
        // during the rewrites in mlir.rs. Leaving them makes the code error prone and not robust

        // we first find the highest old id
        let mut largest_id = 0;
        for (_id, node) in &self.nodes {
            if *node.get_old_id() > largest_id {
                largest_id = *node.get_old_id();
            }
        }

        // we increment to get a new id not existing in the old_id list
        // this gives us a new key to use in the map 
        // for id -> mlir::value used in EqualitySaturationPass.cpp
        largest_id += 1;

        // we initialize a new map which will store the nodes with updated old_id when needed
        // we use an ordered map to keep the program deterministic devoid of randomness
        let mut cleaned_map: BTreeMap<i32, Node> = BTreeMap::new();

        for (id, node) in &self.nodes {
            // We check id the old_id is -1 to see if we need any updates for the node
            if *node.get_old_id() == -1 {
                let new_id = largest_id;
                largest_id += 1;

                let data = node.get_data().to_string();
                let children = node.get_children().to_vec();
                let rows = *node.get_rows();
                let columns = *node.get_columns();
                let mut old_op_id = *node.get_old_op_id();

                println!("Found node with data: {} and dims {}x{} with old_id -1.", data, rows, columns);
                println!("New old_id value: {}", new_id);
                
                // we check if old op id had the same value as old id
                if old_op_id == -1 {
                    // if so we re-assign it to the same value again
                    old_op_id = new_id;
                }

                // insert node with updated fields
                cleaned_map.insert(*id, Node { data, children, rows, columns, old_id: new_id, old_op_id});

            } else {
                // else the old_id value is normal
                // we only need to clone the original node and add it to the map
                cleaned_map.insert(*id, node.clone());
            }
        }

        // read a file and parse it to dot
        // open the file
        let mut file = match File::create(path) {
            Err(why) => panic!("couldn't create file: {}", why),
            Ok(file) => file,
        };

        file.write_all(b"digraph {\n").expect("couldn't write data");


        // construct the file using the new map
        for (id, node) in &cleaned_map {
            let mut rows = node.rows.to_string();
            let mut columns = node.columns.to_string();

            // check if the value of dims is not useful
            // note, only matrices will store accurate 
            // and useful values of the dims at this point
            if rows == "0" && columns == "0" {
                // to make it clear we replace those non useful dims with NA
                rows = "NA".to_string();
                columns = "NA".to_string();
            }

            println!("id: {} with rows: {} and columns: {}", id, rows, columns);

            // the output dot contains additional information corresponding to oldId and oldOpId
            file.write_all(format!("\t{} [label=\"{}\", rows=\"{}\", columns=\"{}\", oldID=\"{}\", oldOpID=\"{}\"];\n", 
                id, node.data, rows, columns, node.old_id, node.old_op_id).as_bytes())
                .expect("couldn't write data");
        }

        file.write_all(b"\n").expect("couldn't write data!");
        
        // now we write the information for the edges
        // for each node, write all children
        for (id, node) in &self.nodes {
            
            for child in node.get_children() {
                file.write_all(format!("\t{id} -> {child};\n").as_bytes())
                    .expect("couldn't write data");
            }
        }

        file.write_all(b"}").expect("couldn't write data");
        // file auto closed
    }
}
