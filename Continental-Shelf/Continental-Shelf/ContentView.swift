//
//  ContentView.swift
//  Continental-Shelf
//
//  Created by 長政輝 on 2023/02/11.
//

import SwiftUI
import simd
import RealityKit
import SceneKit

struct CoralData: Codable {
    var positions: [simd_float3]
    var normals: [simd_float3]
    
    func build() -> ModelEntity {
        var descr = MeshDescriptor()
        descr.positions = MeshBuffers.Positions(positions)
        descr.normals = MeshBuffers.Normals(normals)
        descr.primitives = .triangles([UInt32](0...UInt32(descr.positions.count)))
        let material = SimpleMaterial()
        let model = ModelEntity(mesh: try! .generate(from: [descr]), materials: [material])
        return model
    }
}

struct ContentView: View {
    @State var model = ModelEntity(mesh: .generateBox(size: 1), materials: [SimpleMaterial()])
    
    var body: some View {
        NavigationView {
            OrbitView(entity: model, firstRadius: 6)
                .ignoresSafeArea()
                .toolbar {
                    // リセットボタン
                    ToolbarItem(placement: .navigationBarTrailing) {
                        Button(action: {
                            getData(endpoint: "tetrahedron")
                        }) {
                            Image(systemName: "arrow.clockwise")
                        }
                    }
                    // 成長ボタン
                    ToolbarItem(placement: .bottomBar) {
                        Button(action: {
                            getData(endpoint: "growth")
                        }) {
                            Image(systemName: "goforward.plus")
                                .font(.largeTitle)
                        }
                    }
                }
        }
    }
    
    func getData(endpoint: String) {
        guard let url = URL(string: "http://localhost:8000/" + endpoint) else { return }
        URLSession.shared.dataTask(with: url) {(data, response, error) in
            do {
                if let data = data {
                    let decodedData = try JSONDecoder().decode(CoralData.self, from: data)
                    model = decodedData.build()
                } else {
                    print("No data", data as Any)
                }
            } catch {
                print("Error", error)
            }
        }.resume()
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
