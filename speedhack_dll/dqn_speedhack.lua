-- DQN_deadcells helper, installed into Cheat Engine's autorun folder by speedhack.py.
-- It does nothing unless Cheat Engine was started by speedhack.py (which sets
-- DQN_SPEEDHACK_PID / DQN_SPEEDHACK_SPEED / DQN_SPEEDHACK_OUT), so normal CE use is unaffected.
-- When started that way it turns on CE's own speedhack (autorun/SpeedhackV3.lua) for that
-- process and writes the address of speedhack_wantedspeed to DQN_SPEEDHACK_OUT, so Python
-- can change the speed later by writing that float, exactly like CE's OnSetSpeed does.
local dqn_pid = tonumber(os.getenv("DQN_SPEEDHACK_PID") or "")
if dqn_pid then
  local dqn_speed = tonumber(os.getenv("DQN_SPEEDHACK_SPEED") or "") or 1
  local dqn_out = os.getenv("DQN_SPEEDHACK_OUT")

  local function dqn_report(status, addr, msg)
    if not dqn_out then return end
    local f = io.open(dqn_out .. ".tmp", "w")
    if f then
      f:write(status, "\n", string.format("%X", addr or 0), "\n", tostring(msg or ""), "\n")
      f:close()
      os.rename(dqn_out .. ".tmp", dqn_out)
    end
  end

  local dqn_timer = createTimer(nil, false)
  dqn_timer.Interval = 1000  -- let CE finish starting up first
  dqn_timer.OnTimer = function(t)
    t.Enabled = false
    local ok, err = pcall(function()
      openProcess(dqn_pid)
      if getOpenedProcessID() ~= dqn_pid then
        error("openProcess(" .. dqn_pid .. ") failed")
      end
      speedhack_setSpeed(dqn_speed)
      local a = getAddressSafe("speedhack_wantedspeed")
      if a == nil then
        error("speedhack_wantedspeed not found - CE's speedhack did not start")
      end
      dqn_report("ok", a, "")
    end)
    if not ok then dqn_report("error", 0, err) end
  end
  dqn_timer.Enabled = true
end
