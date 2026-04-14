/**
 * Small top-down 4-wing butterfly for the flutter · friends wordmark (matches loader / reference art).
 */
export function LogoButterflyMark({ className }: { className?: string }) {
  const wingFill = "#f2c4d4";
  const wingStroke = "#b86d88";
  const veinStroke = "#c47a92";
  const bodyStroke = "#4a1f30";
  const bodyFill = "#5c2838";

  return (
    <span className={className} aria-hidden>
      <svg
        className="logo-butterfly-mark-svg"
        viewBox="-38 -24 76 48"
        width={22}
        height={14}
        fill="none"
      >
        <g strokeLinecap="round" strokeLinejoin="round">
          <path
            d="M 0 -8.5 C -9 -15 -33 -14 -31.5 -4.5 C -29 4 -13 7.5 -2.5 5.5 C -0.5 4.5 0 1 0 -8.5 Z"
            fill={wingFill}
            stroke={wingStroke}
            strokeWidth="0.55"
          />
          <path
            d="M 0 3.5 C -17 5.5 -27 12.5 -23.5 17.5 C -17 20.5 -6.5 15.5 0 11 Z"
            fill={wingFill}
            stroke={wingStroke}
            strokeWidth="0.55"
          />
          <path
            d="M 0 -5 Q -14 -10 -24 -5 M 0 -1.5 Q -16 -2 -22 2 M -6 6 Q -14 8 -18 12"
            stroke={veinStroke}
            strokeWidth="0.38"
          />
        </g>
        <g strokeLinecap="round" strokeLinejoin="round">
          <path
            d="M 0 -8.5 C 9 -15 33 -14 31.5 -4.5 C 29 4 13 7.5 2.5 5.5 C 0.5 4.5 0 1 0 -8.5 Z"
            fill={wingFill}
            stroke={wingStroke}
            strokeWidth="0.55"
          />
          <path
            d="M 0 3.5 C 17 5.5 27 12.5 23.5 17.5 C 17 20.5 6.5 15.5 0 11 Z"
            fill={wingFill}
            stroke={wingStroke}
            strokeWidth="0.55"
          />
          <path
            d="M 0 -5 Q 14 -10 24 -5 M 0 -1.5 Q 16 -2 22 2 M 6 6 Q 14 8 18 12"
            stroke={veinStroke}
            strokeWidth="0.38"
          />
        </g>
        <ellipse cx="0" cy="1" rx="1.35" ry="10.5" fill={bodyFill} stroke={bodyStroke} strokeWidth="0.35" />
        <path
          d="M 0 -9.2 Q -1.8 -14 -3.8 -17.2 M 0 -9.2 Q 1.8 -14 3.8 -17.2"
          stroke={bodyStroke}
          strokeWidth="0.65"
        />
        <circle cx="-4.1" cy="-17.4" r="1.05" fill={bodyFill} />
        <circle cx="4.1" cy="-17.4" r="1.05" fill={bodyFill} />
      </svg>
    </span>
  );
}
