import type { Metadata } from "next";
import { TrajectoryDashboard } from "./trajectory-dashboard";

export const metadata: Metadata = {
  title: "RL4RL Architecture Trajectories",
  description: "Compare parameter-count trajectories across research-agent process interventions.",
};

export default function Home() {
  return <TrajectoryDashboard />;
}
