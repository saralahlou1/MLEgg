#include "Support/Graph.h"
#include <llvm/ADT/Hashing.h>
#include <mlir/IR/Value.h>
#include <fstream>
#include <iostream>
#include <string>
#include <regex>

Graph::Node& Graph::add_node_with_dims(int id, std::string data, int rows, int columns, int old_id, int old_op_id) {
    nodes.try_emplace(id, Graph::Node{data, std::vector<int>(), rows, columns, old_id, old_op_id});
    return nodes.find(id)->second;
}

Graph::Node& Graph::add_node(int id, std::string data, int old_id, int old_op_id) {
    nodes.try_emplace(id, Graph::Node{data, std::vector<int>(), 0, 0, old_id, old_op_id});
    return nodes.find(id)->second;
}

void Graph::to_file(const std::string& filename) {
    std::string sep = directed ? " -> " : " -- ";

    // is this the best way to do things?
    // TODO: maybe use a pipe
    std::ofstream file;
    file.open(filename);

    // write the file header
    file << (directed ? "digraph" : "graph") << " {\n";

    // write the nodes
    for (auto const& [id, node] : nodes) {
        file << '\t' << id << " [label=\"" << node.data << "\", rows=" << node.rows << ", columns=" << node.columns << "];\n";
    }

    file << '\n';

    // write the edges
    for (auto const& [id, node] : nodes) {
        for (auto const& other : node.children) {
            file << '\t'<< id << sep << other << ";\n";
        }
    }

    // write the footer
    file << "}\n";

    // clean up
    file.close();
}

Graph Graph::from_file(const std::string& filename) {
    std::ifstream file(filename);

    // error check
    if (!file.is_open()) {
        // error!
        return Graph();
    }
    
    Graph output = Graph();

    // same reading algo:
    // read first line for directedness
    std::string line;

    if (!std::getline(file, line)) {
        // error!
    }
    // assume directed
    output.directed = true;
    // while there isn't a newline: add nodes
    std::regex node_regex("^\t(\\d+) \\[label=\"(.*)\", rows=\"(.*)\", columns=\"(.*)\", oldID=\"(.*)\", oldOpID=\"(.*)\"\\];$");
    std::smatch base_match;
    while (std::getline(file, line) && !line.empty()) {
        // lines look like ([whitespace]<id> [label="<data>"];)
        // c++ doesn't support capture groups :(
        if (std::regex_match(line, base_match, node_regex)) {
            if (base_match.size() > 5) {
                int id = std::stoi(base_match[1].str());
                std::string data = base_match[2].str();
                std::string rows = base_match[3].str();
                std::string columns = base_match[4].str();
                int old_id = std::stoi(base_match[5].str());
                int old_op_id = std::stoi(base_match[6].str());
                std::cout << "id: " << id << ", data: " << data << ", rows: " << rows << ", columns: " << columns << ", old ID: "
                << old_id << ", old op ID: " << old_op_id <<"\n";
                if (rows == "NA" || columns == "NA")
                {
                    std::cout << "It's NA \n";
                    output.add_node(id, data, old_id, old_op_id);
                } else {
                    int row = std::stoi(rows);
                    int column = std::stoi(columns);
                    std::cout << "It's a matrix with " << row << " rows and " << column << " columns \n";
                    output.add_node_with_dims(id, data, row, column, old_id, old_op_id);
                }
            }
        }
    }

    // after the newline: add children
    std::regex edge_regex("^\t(\\d+) -[->] (\\d+);$");
    while (std::getline(file, line) && line != "}") {
        // lines look like ([whitespace]<parent> -[->] <child>;)
        if (std::regex_match(line, base_match, edge_regex)) {
            if (base_match.size() > 2) {
                int from = std::stoi(base_match[1].str());
                int to = std::stoi(base_match[2].str());
                std::cout << "from: " << from << ", to: " << to << "\n";
                auto& from_node = output.nodes.find(from)->second; // not error checked!
                from_node.children.push_back(to);
            }
        }
    }

    file.close();
    
    return output;
}
