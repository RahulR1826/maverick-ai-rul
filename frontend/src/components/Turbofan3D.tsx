import { useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Grid } from '@react-three/drei';
import * as THREE from 'three';

// Decorative 3D turbofan; the colour reflects the engine's predicted risk.
function TurbofanEngine({ color }: { color: string }) {
  const ref = useRef<THREE.Group>(null);
  const fanRef = useRef<THREE.Group>(null);
  
  useFrame((_, delta) => {
    if (fanRef.current) {
      fanRef.current.rotation.z -= delta * 2;
    }
  });

  return (
    <group ref={ref} position={[0, 0, 0]} rotation={[0, -Math.PI / 4, 0]}>
      {/* Nacelle/Cowling (Main Body - Horizontal Cylinder) */}
      <mesh rotation={[Math.PI / 2, 0, 0]}>
        <cylinderGeometry args={[1.8, 1.8, 4.5, 32]} />
        <meshStandardMaterial color="#8BA2B8" metalness={0.6} roughness={0.3} />
      </mesh>
      
      {/* Fan Assembly at Front */}
      <group ref={fanRef} position={[0, 0, 2.25]}>
         {/* Spinner */}
         <mesh rotation={[Math.PI / 2, 0, 0]} position={[0, 0, 0.4]}>
            <coneGeometry args={[0.4, 0.8, 16]} />
            <meshStandardMaterial color="#C0CDDA" metalness={0.7} roughness={0.2} />
         </mesh>

         {/* Fan Blades (12 full-diameter boxes forming 24 spokes) */}
         {Array.from({ length: 12 }).map((_, i) => (
            <mesh key={i} rotation={[0, 0, (i / 12) * Math.PI]}>
               <boxGeometry args={[3.4, 0.08, 0.15]} />
               <meshStandardMaterial color="#AAB8C6" metalness={0.8} roughness={0.4} />
            </mesh>
         ))}
      </group>
      
      {/* Core casing / exhaust at Rear */}
      <mesh rotation={[Math.PI / 2, 0, 0]} position={[0, 0, -2.5]}>
        <cylinderGeometry args={[0.7, 0.5, 1.5, 16]} />
        <meshStandardMaterial color="#566A7D" metalness={0.8} roughness={0.6} />
      </mesh>

      {/* Sensor Nodes (Glowing markers anchored on the cylinder body surface) */}
      {[
        { pos: [1.8, 0, 1.5], label: 'T24' },
        { pos: [-1.8, 0.5, 0], label: 'P30' },
        { pos: [0, 1.8, -1.0], label: 'Nf' },
        { pos: [0, -1.8, 0.5], label: 'T50' },
      ].map((sensor, i) => (
        <group key={i} position={sensor.pos as [number, number, number]}>
          <mesh>
            <sphereGeometry args={[0.08, 16, 16]} />
            <meshStandardMaterial color={color} emissive={color} emissiveIntensity={2} />
          </mesh>
          <pointLight color={color} intensity={0.5} distance={2} />
        </group>
      ))}
    </group>
  );
}

export function Turbofan3D({ color }: { color: string }) {
  return (
    <Canvas camera={{ position: [5, 2, 5], fov: 45 }}>
      <ambientLight intensity={1.5} />
      <directionalLight position={[10, 10, 10]} intensity={2.5} />
      <directionalLight position={[-10, -10, -5]} intensity={2} color="#00BFFF" />
      <pointLight position={[0, 0, 5]} intensity={2.5} color="#ffffff" />
      <Grid infiniteGrid fadeDistance={25} sectionColor="rgba(0,191,255,0.15)" cellColor="rgba(0,191,255,0.15)" sectionThickness={1.5} cellThickness={0.5} position={[0, -2.5, 0]} />
      <TurbofanEngine color={color} />
      <OrbitControls enablePan={false} maxPolarAngle={Math.PI / 1.5} minDistance={3} maxDistance={12} />
    </Canvas>
  );
}
