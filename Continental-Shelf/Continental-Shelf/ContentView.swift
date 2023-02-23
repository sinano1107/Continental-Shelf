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
    var rgb: [CGFloat]
    
    func build() -> ModelEntity {
        var descr = MeshDescriptor()
        descr.positions = MeshBuffers.Positions(positions)
        descr.normals = MeshBuffers.Normals(normals)
        descr.primitives = .triangles([UInt32](0...UInt32(descr.positions.count)))
        let color = UIColor(red: rgb[0], green: rgb[1], blue: rgb[2], alpha: 1)
        let material = SimpleMaterial(color: color, roughness: 1, isMetallic: false)
        let model = ModelEntity(mesh: try! .generate(from: [descr]), materials: [material])
        return model
    }
}

struct ContentView: View {
    @State var model = ModelEntity(mesh: .generateBox(size: 1), materials: [SimpleMaterial()])
    
    var body: some View {
        NavigationView {
            ZStack {
                OrbitView(entity: model, firstRadius: 3)
                    .ignoresSafeArea()
                    .toolbar {
                        // リセットボタン
                        ToolbarItem(placement: .navigationBarTrailing) {
                            Button(action: {
                                getData(endpoint: "generate")
                            }) {
                                Image(systemName: "arrow.clockwise")
                            }
                        }
                }
                VStack {
                    Spacer()
                    HStack {
                        Button(action: {
                            getData(endpoint: "update/false")
                        }) {
                            Text("👎")
                                .font(.largeTitle)
                        }
                        Spacer()
                        Button(action: {
                            getData(endpoint: "update/true")
                        }) {
                            Text("👍")
                                .font(.largeTitle)
                        }
                    }
                    .padding(.all)
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
